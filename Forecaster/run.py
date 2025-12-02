from model import LSTMModel
import logging
import torch
import pandas as pd
import numpy as np
import os
import ast
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from data_loader import load_data, create_windows
from metrics_evaluation import calculate_metrics
from model import *
from datetime import datetime
from joblib import dump, load
from torch.optim import AdamW
from torch.optim import lr_scheduler
from statsmodels.tsa.arima.model import ARIMAResults
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Forecaster:
    def __init__(self, config):
        """Initialize forecaster with configuration"""
        self.config = config
        self.model = None
        self.data = None
        self.test_max = None
        self.test_min = None
        self.forecast_max = None
        self.forecast_min = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process_sql_features(self, sql_feature_df, output_path):
        sql_feature_df['id'] = pd.factorize(sql_feature_df['template_id'])[0] + 1  
        sql_feature_df['feature_vector'] = sql_feature_df['feature_vector'].apply(lambda x: ast.literal_eval(x))
        sql_feature_df['feature_vector'] = sql_feature_df['feature_vector'].apply(lambda x: np.array(x))
        feature_matrix = np.stack(sql_feature_df['feature_vector'].values)
        feature_df = pd.DataFrame(feature_matrix, columns=[f'feature_{i + 1}' for i in range(feature_matrix.shape[1])])
        id_columns = sql_feature_df[['template_id', 'id']]
        final_df = pd.concat([id_columns, feature_df], axis=1)
        final_df.to_csv(os.path.join(output_path, "processed_sql_features.csv"), index=False)


    def build_model(self,scaler):
        model_type = self.config.get('model_type', 'LSTMModel')
        model_class = globals().get(model_type)
        if model_type not in ['RandomForest', 'HistoryMean', 'NaiveRepeat','ARIMAModel', 'QB5000', 'WFGAN'] and model_class is None:
            raise ValueError(f"Unsupported model: {model_type}")
        model_params = {
            'input_dim': self.config['input_dim'],
            'output_dim': self.config['output_dim'],
            'window_size': self.config['window_size'],
            'prediction_length': self.config['prediction_length']
        }
        self.model = model_class(**model_params)
        self.model.to(self.device)
        self.model_type = 'torch'
        logger.info(
            f"Initial {model_type}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Input Dim: {self.config['input_dim']:>12}\n"
            f"• Outout Dim: {self.config['output_dim']:>11}\n"
            f"• Window Size：{self.config['window_size']:>10}\n"
            f"• Pre Length：{self.config['prediction_length']:>9}"
        )
    def load_model(self):
        save_dir = "saved_models"
        model_path = os.path.join(
            save_dir,
            f"{self.config['data_set']}_{self.config['model_type']}_{self.config['window_size']}_{self.config['prediction_length']}.pth"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}. Train first.")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        logger.info(
            f"Loaded {self.config['model_type']} from {model_path} "
            f"(epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f})"
        )
    @staticmethod
    def flatten_targets(batch_y):  # [B,H,D] -> [B, H*D]
        B, H, D = batch_y.shape
        return batch_y.reshape(B, H * D)
    def train_model(self):

        if self.model_type == 'torch_static':
            logger.info("HistoryMean Skip Training。")
            return
        logger.info("\n=== Training Dev Information ===")
        logger.info(f"Choose Dev: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"Current CUDA Dev: {torch.cuda.get_device_name(0)}")
        # ====== Fast DataLoader ======
        cpu_workers = max(2, os.cpu_count() // 2)
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=cpu_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        try:
            from tqdm import tqdm as _tqdm
            _tqdm.get_lock()
        except Exception:
            pass
        # === optimizer & scheduler ===
        base_lr = float(self.config.get('learning_rate', 1e-3))
        extractor_lr_scale = float(self.config.get('extractor_lr_scale', 0.1))
        backbone_params, extractor_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if 'extractors' in name:
                extractor_params.append(p)
            else:
                backbone_params.append(p)
        if len(extractor_params) == 0:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=base_lr, weight_decay=1e-4)
        else:
            optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": base_lr, "weight_decay": 1e-4},
                    {"params": extractor_params, "lr": base_lr * extractor_lr_scale, "weight_decay": 1e-4},
                ]
            )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-8
        )
        print(f"[OptGroups] backbone={len(backbone_params)} tensors, extractors={len(extractor_params)} tensors, "
              f"lr_backbone={base_lr:.2e}, lr_extractors={base_lr * extractor_lr_scale:.2e}")
        use_adaptive_tw = (self.config.get("data_type", "") == "hyper")
        if use_adaptive_tw:
            resource_dim = int(self.config.get("resource_dim", 6))
            tw_alpha = float(self.config.get("gradnorm_alpha", 0.5))  # 0.25 / 0.5 / 0.75 
            tw_momentum = float(self.config.get("task_ema_momentum", 0.9))
            tw_eps = 1e-6
            ema_sql = None
            ema_res = None
        early_stop_patience = 5
        min_delta = self.config['early_stopping']
        best_loss = float('inf')
        no_improve_count = 0
        class CharbonnierLoss(nn.Module):
            def __init__(self, eps=1e-6):
                super().__init__()
                self.eps = eps
            def forward(self, pred, target):
                return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps))
        criterion = CharbonnierLoss(eps=1e-6)
        save_dir = "saved_models"
        os.makedirs(save_dir, exist_ok=True)
        for epoch in tqdm(range(self.config['epochs']), desc="Epoch Progress"):
            self.model.train()
            total_loss = 0.0
            batch_idx = 0
            for batch in train_loader:
                batch_X, batch_y = batch
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if use_adaptive_tw:
                    sql_target = batch_y[..., :-resource_dim]
                    res_target = batch_y[..., -resource_dim:]
                    model_out = self.model(batch_X)
                    if isinstance(model_out, (tuple, list)):
                        if len(model_out) == 3:
                            sql_out, res_out, bl_loss = model_out
                        elif len(model_out) == 2:
                            outputs, bl_loss = model_out
                            sql_out = outputs[..., :-resource_dim]
                            res_out = outputs[..., -resource_dim:]
                        else:
                            outputs = model_out[0]
                            sql_out = outputs[..., :-resource_dim]
                            res_out = outputs[..., -resource_dim:]
                            bl_loss = torch.tensor(0.0, device=self.device)
                    else:
                        outputs = model_out
                        sql_out = outputs[..., :-resource_dim]
                        res_out = outputs[..., -resource_dim:]
                        bl_loss = torch.tensor(0.0, device=self.device)

                    loss_sql = criterion(sql_out, sql_target)
                    loss_res = criterion(res_out, res_target)
                    with torch.no_grad():
                        if ema_sql is None:
                            ema_sql = loss_sql.detach()
                            ema_res = loss_res.detach()
                            logger.info(f"Initial task losses - SQL: {float(ema_sql):.4f}, "
                                        f"Resource: {float(ema_res):.4f}")
                        else:
                            ema_sql = tw_momentum * ema_sql + (1.0 - tw_momentum) * loss_sql.detach()
                            ema_res = tw_momentum * ema_res + (1.0 - tw_momentum) * loss_res.detach()
                    inv_sql = (ema_sql + tw_eps).pow(-tw_alpha)
                    inv_res = (ema_res + tw_eps).pow(-tw_alpha)
                    inv_sum = inv_sql + inv_res + tw_eps
                    w_sql = 2.0 * inv_sql / inv_sum
                    w_res = 2.0 * inv_res / inv_sum
                    main_loss = w_sql * loss_sql + w_res * loss_res + bl_loss
                    main_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    optimizer.step()
                    total_loss += main_loss.item()
                    batch_idx += 1
                    if (batch_idx % 500) == 0:
                        logger.info(
                            f"[TW] w_sql={float(w_sql):.3f} w_res={float(w_res):.3f} "
                            f"L_sql={float(loss_sql.detach()):.3f} L_res={float(loss_res.detach()):.3f}"
                        )
                else:
                    if self.config['model_type'] in [
                        'PathFormer', 'QuadraFormer', 'QuadraFormer_star',
                        'QuadraFormer_woc', 'QuadraFormer_woa', 'QuadraFormer_wos'
                    ]:
                        outputs, bl_loss = self.model(batch_X)
                        loss = criterion(outputs, batch_y) + bl_loss.to(self.device)
                    else:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()
                    batch_idx += 1
            avg_loss = total_loss / max(1, batch_idx)
            logger.info(f"Epoch {epoch + 1}: Training Loss {avg_loss:.4f}")
            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Current Learning Rate: {current_lr:.6e}")
            if avg_loss < best_loss - min_delta and avg_loss != 0:
                best_loss = avg_loss
                no_improve_count = 0
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'loss': avg_loss,
                }, os.path.join(
                    save_dir,
                    f"{self.config['data_set']}_{self.config['model_type']}_"
                    f"{self.config['window_size']}_{self.config['prediction_length']}.pth"
                ))
            else:
                no_improve_count += 1
                if no_improve_count >= early_stop_patience:
                    logger.info(f"Early stopping triggered after {early_stop_patience} epochs without improvement.")
                    break
    def evaluate(self, dict_df, column_labels):
        logger.info("\n=== Dev information ===")
        logger.info(f"Choose Dev: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"Current CUDA ID: {torch.cuda.current_device()}")
            logger.info(f"Dev Name: {torch.cuda.get_device_name(0)}")
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=self.config['batch_size'],
                                 shuffle=False)
        self.model.eval()
        all_outputs, all_targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                if self.config['model_type'] in ['PathFormer', 'QuadraFormer', 'QuadraFormer_star', 'QuadraFormer_woc', 'QuadraFormer_woa', 'QuadraFormer_wos']:
                    outputs, _ = self.model(batch_X)
                else:
                    outputs = self.model(batch_X)
                all_outputs.append(outputs)
                all_targets.append(batch_y)
        preds = torch.cat(all_outputs, dim=0).cpu()
        preds = torch.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)
        targets = torch.cat(all_targets, dim=0).cpu()
        metrics = calculate_metrics(preds,
                                    targets,
                                    self.config,
                                    dict_df,
                                    column_labels,
                                    getattr(self, 'scaler', None),
                                    self.test_max,
                                    self.test_min,
                                     )
        return (preds, metrics) if self.config['mode'] == 'forecast' else (0.0, metrics)

    def forecast(self, dict_df, column_labels):
        logger.info("\n=== Forecast Information ===")
        logger.info(f"Choose Dev: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"Current CUDA ID: {torch.cuda.current_device()}")

        forecast_loader = DataLoader(self.forecast_dataset,
                                     batch_size=self.config['batch_size'],
                                     shuffle=False)
        self.model.eval()
        all_outputs, all_targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in forecast_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                if self.config['model_type'] in ['PathFormer', 'QuadraFormer', 'QuadraFormer_star', 'QuadraFormer_woc', 'QuadraFormer_woa', 'QuadraFormer_wos']:
                    outputs, _ = self.model(batch_X)
                else:
                    outputs = self.model(batch_X)
                all_outputs.append(outputs)
                all_targets.append(batch_y)
        preds = torch.cat(all_outputs, dim=0).cpu()  # (N , H , D)
        preds = torch.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)  
        targets = torch.cat(all_targets, dim=0).cpu()
        if self.config["QPS"] != "True":
            metrics = calculate_metrics(preds, targets,
                                    self.config,
                                    dict_df,
                                    column_labels,
                                    getattr(self, "scaler", None),  # 若有 scaler 就用
                                    self.test_max,
                                    self.test_min,
                                    )

            return preds, targets, metrics
        else:
            N, H, D = preds.shape
            pred_flat = preds.detach().cpu().numpy().reshape(N * H, D)
            target_flat = targets.detach().cpu().numpy().reshape(N * H, D)
            pred_inv = self.scaler.inverse_transform(pred_flat)
            target_inv = self.scaler.inverse_transform(target_flat)
            df = pd.DataFrame({
                'predicted_qps': pred_inv[:, 0],
                'true_qps': target_inv[:, 0]
            })
            os.makedirs("rst/qps", exist_ok=True)
            out_path = f"rst/qps/qps_prediction_{config['window_size']}.csv"
            df.to_csv(out_path, index=False)
            print(f"Saved detailed QPS prediction results to {out_path}")
            return None, None, None

    def save_results(self):
        """Save forecast results"""
        # Add result saving code here
        pass

    def run(self):
        logger.info("Start...")
        self.train_dataset, self.test_dataset, self.forecast_dataset, \
            scaler, dict_df, column_labels, \
            self.test_max, self.test_min, self.forecast_max, self.forecast_min = load_data(self.config)
        self.scaler = scaler
        self.build_model(scaler)
        mode = self.config['mode']
        # ----- Train -----
        if mode == 'train':
            logger.info(f"Build New Model:\n{self.model}")
            self.train_model()
            if self.config["QPS"] != "True":
                _, metrics = self.evaluate(dict_df, column_labels)
            else:
                _, _, metrics = self.forecast(dict_df, column_labels)
        # ----- Test -----
        elif mode == 'test':
            if self.model_type not in ['sklearn', 'torch_static']:
                self.load_model()
            _, metrics = self.evaluate(dict_df, column_labels)
        # ----- Forecast -----
        elif mode == 'forecast':
            if self.model_type not in ['sklearn', 'torch_static']:
                self.load_model()
            _, _, metrics = self.forecast(dict_df, column_labels)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        logger.info(f"Evaluation metrics: {metrics} on {self.config['data_set']}")


# Example usage:
config = {
    "mode":"test",             #[train,test,forecast]
    "interval":1,             #[None,1,6,12,24]qps:48
    "data_set":"SDSS",          #[SDSS,tiramisu,tiramisu-sim,alibaba]
    "data_path": "processed\\SDSS\\0.1sampling",  # Data path[0.1sampling]
    "data_type": "hyper",          #[hyper,sql,resource]
    "model_type": "QuadraFormer_star",      #[HistoryMean,QB5000,WFGAN,Stacked_LSTM,Transformer,PathFormer,QuadraFormer,QuadraFormer_star]
                                #  NaiveRepeat,ARIMAModel,RandomForest
    "processed":"True",         #  [True, Flase]
    "all_col":'True',           #  [True, Flase]
    "less":'Flase',              #  [True, Flase]
    "QPS":"Flase",           #  [True, Flase]
    "window_size": 32,
    "prediction_length": 32 ,
    "batch_size": 16,
    "learning_rate": 2e-4,      #2e-4  SDSS 1E-3
    "early_stopping": 1e-3,
    "epochs": 10,
    "input_dim": None,
    "output_dim": None,
    "tolerance": 1.5,
    "adaptive_patch": "true",
    "patch_d_min": 1,
    "patch_d_max": 16,  #4,10
    "embed_variant": "gnn",     # ['gnn','rand','avgw2v']
}

if __name__ == '__main__':
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    torch.backends.cudnn.benchmark = True
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_training.log"
    logging.basicConfig(
        filename=log_dir + log_filename,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    logger = logging.getLogger()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)
    import multiprocessing
    multiprocessing.freeze_support()
    forecaster = Forecaster(config)
    forecaster.run()

