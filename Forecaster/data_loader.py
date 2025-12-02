import os
import ast
import numpy as np
import pandas as pd
import torch
from datetime import timedelta
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler


def create_windows(data, config):
    X, y = [], []
    window_size = config['window_size']
    pred_length = config['prediction_length']

    for i in range(len(data) - window_size - pred_length + 1):
        X.append(data.iloc[i:i + window_size, :].values)
        y.append(data.iloc[i + window_size:i + window_size + pred_length, :].values)
    X = np.array(X)  # (n_samples, window_size, n_features)
    y = np.array(y)  # (n_samples, pred_length, n_features)

    if y.ndim != X.ndim:
        print(f"Dim warning: X.ndim={X.ndim}, y.ndim={y.ndim}, extend y...")
        while y.ndim < X.ndim:
            y = np.expand_dims(y, axis=-1)
            print(f"Shape: y {y.shape}")
    return X, y


def create_nonoverlap_windows(data, config):
    X, y = [], []
    window_size = config['window_size']
    pred_length = config['prediction_length']
    step = window_size  

    for i in range(0, len(data) - window_size - pred_length + 1, step):
        X.append(data.iloc[i:i + window_size, :].values)
        y.append(data.iloc[i + window_size:i + window_size + pred_length, :].values)
    X = np.array(X)
    y = np.array(y)

    if y.ndim != X.ndim:
        print(f"Dim warning: X.ndim={X.ndim}, y.ndim={y.ndim}，extend y...")
        while y.ndim < X.ndim:
            y = np.expand_dims(y, axis=-1)
            print(f"Shape: y {y.shape}")
    return X, y


def load_data(config, scaler=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = config.get('data_path')
    original_data_path = os.path.join(project_root, data_path)

    window_subdir = f"window_{config['window_size']}_{config['prediction_length']}"
    absolute_data_path = os.path.join(original_data_path, window_subdir)
    print(f"Using data path: {absolute_data_path}")
    os.makedirs(absolute_data_path, exist_ok=True)

    if config.get("QPS", "False") == "True":
        timestamps_file = os.path.join(original_data_path, "timestamps.csv")
        df_ts = pd.read_csv(timestamps_file)
        df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'], unit='s')
        qps_freq = config.get("QPS_FREQ", "min").lower() 
        df_ts['timestamp'] = df_ts['timestamp'].dt.floor(qps_freq)
        qps_series = df_ts.groupby('timestamp').size().rename('qps').reset_index()
        vals = qps_series['qps'].values.astype(np.float32)
        W, H = config['window_size'], config['prediction_length']
        X_all, y_all = [], []
        for i in range(len(vals) - W - H + 1):
            X_all.append(vals[i:i + W])
            y_all.append(vals[i + W:i + W + H])
        X_all = np.array(X_all)[:, :, None]  # (N_all, W, 1)
        y_all = np.array(y_all)[:, :, None]  # (N_all, H, 1)
        N_all = len(X_all)
        split = int(0.8 * N_all)
        X_train, y_train = X_all[:split], y_all[:split]
        X_test, y_test = X_all[split:], y_all[split:]

        fps = 60 if config.get("QPS_FREQ", "min").lower() in ('min', 't') else 3600
        interval = int(config.get('interval', 0))  # 小时
        vals = qps_series['qps'].values.astype(np.float32)
        last_segment = vals[-interval * fps:]
        pad = np.full(W, last_segment[0], dtype=np.float32)
        padded = np.concatenate([pad, last_segment])
        num_windows = (interval * fps) // H
        X_forc = np.zeros((num_windows, W, 1), dtype=np.float32)
        y_forc = np.zeros((num_windows, H, 1), dtype=np.float32)
        for n in range(num_windows):
            start = n * H
            X_forc[n, :, 0] = padded[start: start + W]
            y_forc[n, :, 0] = padded[start + W: start + W + H]

        scaler = StandardScaler().fit(X_train.reshape(-1, 1))
        def scale(arr):
            s0, s1, s2 = arr.shape
            flat = arr.reshape(-1, s2)
            flat = scaler.transform(flat)
            return flat.reshape(s0, s1, s2)
        X_train = scale(X_train); X_test = scale(X_test); X_forc = scale(X_forc)
        y_train = scale(y_train); y_test = scale(y_test); y_forc = scale(y_forc)

        all_columns = ["qps"]
        np.save(os.path.join(absolute_data_path, "columns.npy"), np.array(all_columns, dtype=object))
        config['all_columns'] = all_columns

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.float32))
        test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                torch.tensor(y_test, dtype=torch.float32))
        forc_ds = TensorDataset(torch.tensor(X_forc, dtype=torch.float32),
                                torch.tensor(y_forc, dtype=torch.float32))

        dict_df = None
        column_label = None
        test_max = float(y_test.max()); test_min = float(y_test.min())
        forecast_max = float(y_forc.max()); forecast_min = float(y_forc.min())
        config['input_dim'] = 1; config['output_dim'] = 1

        return (
            train_ds, test_ds, forc_ds, scaler,
            dict_df, column_label,
            test_max, test_min, forecast_max, forecast_min
        )

    if config["data_type"] == 'resource':
        required_files = ['X_train.npy', 'y_train.npy', 'X_test.npy', 'y_test.npy']
    else:
        required_files = ['X_train.npy', 'y_train.npy', 'X_test.npy', 'y_test.npy', 'dict.npy', 'column_label.npy']
    if config["interval"] != 'None':
        required_files.extend([f'X_{config["interval"]}h.npy', f'y_{config["interval"]}h.npy'])
    file_paths = {f: os.path.join(absolute_data_path, f) for f in required_files}
    all_files_exist = all(os.path.exists(fp) for fp in file_paths.values())

    scaler = StandardScaler()

    if all_files_exist or config["processed"]=='True':
        print("Loading preprocessed numpy files...")
        if config['less'] == 'True':
            X_train = np.load(file_paths['X_train.npy'])
            y_train = np.load(file_paths['y_train.npy'])
            indices = np.arange(0, X_train.shape[0], 10)
            X_train = X_train[indices]; y_train = y_train[indices]
        else:
            X_train = np.load(file_paths['X_train.npy'])
            y_train = np.load(file_paths['y_train.npy'])

        X_test = np.load(file_paths['X_test.npy'])
        y_test = np.load(file_paths['y_test.npy'])
        if config['interval'] != 'None':
            X_forecast = np.load(file_paths[f"X_{config['interval']}h.npy"])
            y_forecast = np.load(file_paths[f"y_{config['interval']}h.npy"])
        else:
            X_forecast = X_test[:1].copy()
            y_forecast = y_test[:1].copy()

        dict_df = None; column_label = None
        if config["data_type"] != 'resource':
            dict_df = np.load(file_paths['dict.npy'], allow_pickle=True)
            column_label = np.load(file_paths['column_label.npy'], allow_pickle=True)

        test_max = np.max(y_test.reshape(-1, y_test.shape[-1]), axis=0)
        test_min = np.min(y_test.reshape(-1, y_test.shape[-1]), axis=0)
        forecast_max = np.max(y_forecast.reshape(-1, y_forecast.shape[-1]), axis=0)
        forecast_min = np.min(y_forecast.reshape(-1, y_forecast.shape[-1]), axis=0)

        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        X_forecast = scaler.transform(X_forecast.reshape(-1, X_forecast.shape[-1])).reshape(X_forecast.shape)
        y_train = scaler.transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
        y_test = scaler.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)
        y_forecast = scaler.transform(y_forecast.reshape(-1, y_forecast.shape[-1])).reshape(y_forecast.shape)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        X_forecast_tensor = torch.tensor(X_forecast, dtype=torch.float32)
        y_forecast_tensor = torch.tensor(y_forecast, dtype=torch.float32)

        config['input_dim'] = X_train.shape[-1]
        config['output_dim'] = y_train.shape[-1]

        columns_file = os.path.join(absolute_data_path, "columns.npy")
        if os.path.exists(columns_file):
            all_columns = np.load(columns_file, allow_pickle=True).tolist()
        else:
            all_columns = [f"col_{i}" for i in range(X_train.shape[-1])]
        config['all_columns'] = all_columns

        return (
            TensorDataset(X_train_tensor, y_train_tensor),
            TensorDataset(X_test_tensor, y_test_tensor),
            TensorDataset(X_forecast_tensor, y_forecast_tensor),
            scaler,
            dict_df,
            column_label,
            test_max,
            test_min,
            forecast_max,
            forecast_min
        )

    print("Preprocessing data from CSV files...")
    timestamps_file = os.path.join(original_data_path, "timestamps.csv")
    sql_feature_file = os.path.join(original_data_path, "sql_features_modified.csv")
    other_data_file = os.path.join(original_data_path, "other_data.csv")
    dict_file = os.path.join(original_data_path, "template_param_string_dict_modified.csv")
    column_label_file = os.path.join(original_data_path, "feature_label.csv")
    dict_df = None; column_label = None
    try:
        timestamps_df = pd.read_csv(timestamps_file)
        if config["data_type"] in ('sql', 'hyper'):
            sql_feature_df = pd.read_csv(sql_feature_file)
            dict_df = pd.read_csv(dict_file)
            column_label = pd.read_csv(column_label_file)
        if config["data_type"] in ('resource', 'hyper'):
            other_data_df = pd.read_csv(other_data_file)
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
        raise

    timestamps_df['timestamp'] = pd.to_datetime(timestamps_df['timestamp'], unit='s')
    timestamps_df['YY']   = timestamps_df['timestamp'].dt.year
    timestamps_df['MM']   = timestamps_df['timestamp'].dt.month
    timestamps_df['DD']   = timestamps_df['timestamp'].dt.day
    timestamps_df['HOUR'] = timestamps_df['timestamp'].dt.hour
    timestamps_df['MIN']  = timestamps_df['timestamp'].dt.minute
    timestamps_df['SEC']  = timestamps_df['timestamp'].dt.second
    timestamps_df = timestamps_df.drop(columns=['timestamp'])

    def load_template_embeddings(original_data_path: str,
                                 dict_file: str,
                                 variant: str,
                                 embed_file_override: str = ""):
        import os, numpy as np, pandas as pd
        default_map = {
            "gnn":    ("template_gnn_embeddings.csv",    "template_gnn_embeddings.npy"),
            "rand":   ("template_rand_embeddings.csv",   "template_rand_embeddings.npy"),
            "avgw2v": ("template_avgw2v_embeddings.csv", "template_avgw2v_embeddings.npy"),
        }
        if variant not in default_map and not embed_file_override:
            return {}, 0

        cand_csv, cand_npy = default_map.get(variant, ("", ""))
        if embed_file_override:
            if embed_file_override.lower().endswith(".csv"):
                cand_csv, cand_npy = embed_file_override, ""
            elif embed_file_override.lower().endswith(".npy"):
                cand_csv, cand_npy = "", embed_file_override
            else:
                cand_csv, cand_npy = embed_file_override + ".csv", embed_file_override + ".npy"

        csv_path = os.path.join(original_data_path, cand_csv) if cand_csv else ""
        npy_path = os.path.join(original_data_path, cand_npy) if cand_npy else ""
        if csv_path and os.path.isfile(csv_path):
            emb_df = pd.read_csv(csv_path)
            if "template_id" not in emb_df.columns:
                raise ValueError(f"CSV {csv_path} 缺少 template_id 列")
            vec_cols = [c for c in emb_df.columns if c != "template_id"]
            emb_mat = emb_df[vec_cols].to_numpy(dtype=np.float32)
            emb_dim = emb_mat.shape[1]
            dict_df_local = pd.read_csv(dict_file)
            tid_to_num = dict_df_local.assign(
                template_id=dict_df_local["template_id"].astype(str).str.strip()
            ).set_index("template_id")["Num"].to_dict()
            template2embed = {}
            miss, dup = 0, 0
            seen = set()
            for i, tid in enumerate(emb_df["template_id"].astype(str).str.strip().tolist()):
                num = tid_to_num.get(tid, None)
                if num is None:
                    miss += 1
                    continue
                if num in seen:
                    dup += 1
                    continue
                template2embed[int(num)] = emb_mat[i]
                seen.add(num)
            if miss > 0:
                print(f"[Embed] {miss} template_id(s) from {os.path.basename(csv_path)} not found in dict; skipped.")
            if dup > 0:
                print(f"[Embed] {dup} duplicate Num mapping(s) detected; kept first occurrence.")
            return template2embed, emb_dim
    if str(config.get('data_set', '')).lower() != "alibaba":
        dict_df = pd.read_csv(dict_file)
        tid_map = dict_df.set_index('template_id')['Num']  # template_id -> Num
        template_id_series = sql_feature_df['template_id'].map(tid_map)
        missing = template_id_series.isna().sum()
        if missing > 0:
            print(f"[WARN] {missing} template_id(s) not found in dict. Dropping those rows.")
            ok_mask = template_id_series.notna()
            sql_feature_df = sql_feature_df.loc[ok_mask].reset_index(drop=True)
            other_data_df  = other_data_df.loc[ok_mask].reset_index(drop=True)
            timestamps_df  = timestamps_df.loc[ok_mask].reset_index(drop=True)
            template_id_series = template_id_series.loc[ok_mask].reset_index(drop=True)
        template_id_series = template_id_series.astype(int)
        existing_embed_cols = [c for c in sql_feature_df.columns if str(c).startswith("embed_")]
        sql_feature_df_numeric = sql_feature_df.drop(columns=['template_id'] + existing_embed_cols)
        ts_shifted = timestamps_df.groupby(template_id_series).shift(1)
        ts_diff    = (ts_shifted - timestamps_df).fillna(0)
        ts_diff.columns = [c + "_val" for c in ts_diff.columns]
        timestamps_result_df = pd.concat([timestamps_df, ts_diff], axis=1)
        dfs = []
        data_type = str(config.get("data_type","")).lower()
        if data_type in ('sql', 'hyper'):
            sql_shifted = sql_feature_df_numeric.groupby(template_id_series).shift(1)
            sql_diff    = (sql_shifted - sql_feature_df_numeric).fillna(0)
            sql_diff.columns = [c + "_val" for c in sql_diff.columns]
            sql_result_df = pd.concat([sql_feature_df_numeric, sql_diff], axis=1)
            dfs.append(sql_result_df)
        if data_type in ('resource', 'hyper'):
            other_shifted = other_data_df.groupby(template_id_series).shift(1)
            other_diff    = (other_shifted - other_data_df).fillna(0)
            other_diff.columns = [c + "_val" for c in other_diff.columns]
            other_result_df = pd.concat([other_data_df, other_diff], axis=1)
            dfs.append(other_result_df)

        embed_variant  = str(config.get("embed_variant", "gnn")).lower()
        embed_override = str(config.get("embed_file_override", "")).strip()
        embed_df = None

        if existing_embed_cols:
            print(f"[Data] Detected existing embeddings in csv: {len(existing_embed_cols)} dims. Skip re-adding.")
            # embed_df = sql_feature_df[existing_embed_cols].copy()
        else:
            template2embed, emb_dim = load_template_embeddings(
                original_data_path=original_data_path,
                dict_file=dict_file,
                variant=embed_variant,
                embed_file_override=embed_override
            )
            if template2embed and emb_dim > 0:
                zero = np.zeros(emb_dim, dtype=np.float32)
                embed_array = np.vstack([template2embed.get(int(tid), zero) for tid in template_id_series])
                embed_df = pd.DataFrame(
                    embed_array,
                    index=template_id_series.index,
                    columns=[f"embed_{i}" for i in range(emb_dim)]
                )
            else:
                print("[Data] No template embeddings loaded; continue without embed_*.")

        merged_parts = [template_id_series.rename('template_id').to_frame()]
        if embed_df is not None:
            merged_parts.append(embed_df)
        merged_parts.append(timestamps_result_df)
        merged_parts.extend(dfs)

        merged_df = pd.concat(merged_parts, axis=1)
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()].copy()

        embed_cols_in_merged = [c for c in merged_df.columns if str(c).startswith("embed_")]
        print(f"[Data] embed dims in merged_df: {len(embed_cols_in_merged)}")

    else:
        col_name = other_data_df.columns[0]
        other_data_df[col_name] = other_data_df[col_name].str.extract(r'm_(\d+)').astype(int)
        merged_df = other_data_df

    all_columns = merged_df.columns.tolist()
    np.save(os.path.join(absolute_data_path, "columns.npy"), np.array(all_columns, dtype=object))
    config['all_columns'] = all_columns
    resource_dim = int(config.get('resource_dim', 6))
    print("Tail columns (should be resource metrics):", all_columns[-resource_dim:])

    split_idx = int(0.8 * len(merged_df))
    train_df = merged_df[:split_idx]
    test_df = merged_df[split_idx - config['window_size'] - config['prediction_length'] + 1:] 
    X_train, y_train = create_windows(train_df, config)
    X_test, y_test = create_nonoverlap_windows(test_df, config)
    config['input_dim'] = X_train.shape[-1]
    config['output_dim'] = y_train.shape[-1]

    time_intervals = [1, 6, 12, 24, 48]
    if config['data_set'] != "alibaba":
        ts_series = pd.to_datetime(
            merged_df[['YY', 'MM', 'DD', 'HOUR', 'MIN', 'SEC']].rename(
                columns={'YY': 'year', 'MM': 'month', 'DD': 'day',
                         'HOUR': 'hour', 'MIN': 'minute', 'SEC': 'second'}
            ),
            errors='coerce'
        )
    else:
        ts_series = pd.to_datetime(
            merged_df[['YY', 'MM', 'DD', 'HH', 'MIN', 'SEC']].rename(
                columns={'YY': 'year', 'MM': 'month', 'DD': 'day',
                         'HH': 'hour', 'MIN': 'minute', 'SEC': 'second'}
            ),
            errors='coerce'
        )
    end_time = ts_series.max()

    forecast_X, forecast_y = None, None
    interval_cfg_raw = config.get('interval', None)
    try:
        interval_cfg = int(interval_cfg_raw) if interval_cfg_raw not in (None, 'None') else 48
    except Exception:
        interval_cfg = 48

    for interval in time_intervals:
        start_time = end_time - timedelta(hours=interval)
        mask = (ts_series > start_time) & (ts_series <= end_time)
        filtered_data = merged_df.loc[mask].reset_index(drop=True)

        window_size = int(config['window_size'])
        num_rows = int(filtered_data.shape[0])
        if num_rows == 0:
            continue

        num_complete_windows = num_rows // window_size

        if num_complete_windows == 0:
            pad_n = window_size - num_rows
            tail = filtered_data.tail(1).copy()
            filtered_data = pd.concat([filtered_data, pd.concat([tail] * pad_n, ignore_index=True)], ignore_index=True)
            num_complete_windows = 1
        else:
            keep_len = num_complete_windows * window_size
            filtered_data = filtered_data.iloc[-keep_len:].reset_index(drop=True)

        X_interval, Y_interval = create_nonoverlap_windows(filtered_data, config)

        np.save(os.path.join(absolute_data_path, f'X_{interval}h.npy'), X_interval)
        np.save(os.path.join(absolute_data_path, f'y_{interval}h.npy'), Y_interval)
        if interval == interval_cfg:
            forecast_X = X_interval
            forecast_y = Y_interval
    def _load_xy(tag):
        xp = os.path.join(absolute_data_path, f'X_{tag}.npy')
        yp = os.path.join(absolute_data_path, f'y_{tag}.npy')
        return (np.load(xp), np.load(yp)) if (os.path.exists(xp) and os.path.exists(yp)) else (None, None)
    if forecast_X is None or forecast_y is None:
        X48, y48 = _load_xy('48h')
        if X48 is not None:
            forecast_X, forecast_y = X48, y48
        else:
            X1, y1 = _load_xy('1h')
            if X1 is not None:
                forecast_X, forecast_y = X1, y1
            else:
                fx, fy = create_nonoverlap_windows(test_df, config)
                if fx is not None and fy is not None and fx.size > 0 and fy.size > 0:
                    forecast_X, forecast_y = fx, fy
                else:
                    forecast_X = X_test[:1].copy()
                    forecast_y = y_test[:1].copy()
    np.save(os.path.join(absolute_data_path, 'X_train.npy'), X_train)
    np.save(os.path.join(absolute_data_path, 'y_train.npy'), y_train)
    np.save(os.path.join(absolute_data_path, 'X_test.npy'), X_test)
    np.save(os.path.join(absolute_data_path, 'y_test.npy'), y_test)
    if config['data_type'] != 'resource':
        np.save(os.path.join(absolute_data_path, 'dict.npy'), dict_df)
        np.save(os.path.join(absolute_data_path, 'column_label.npy'), column_label)
    test_max = np.max(y_test.reshape(-1, y_test.shape[-1]), axis=0)
    test_min = np.min(y_test.reshape(-1, y_test.shape[-1]), axis=0)
    forecast_max = np.max(forecast_y.reshape(-1, forecast_y.shape[-1]), axis=0)
    forecast_min = np.min(forecast_y.reshape(-1, forecast_y.shape[-1]), axis=0)
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    X_forecast = scaler.transform(forecast_X.reshape(-1, forecast_X.shape[-1])).reshape(forecast_X.shape)
    y_train = scaler.transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
    y_test = scaler.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)
    y_forecast = scaler.transform(forecast_y.reshape(-1, forecast_y.shape[-1])).reshape(forecast_y.shape)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    X_forecast_tensor = torch.tensor(X_forecast, dtype=torch.float32)
    y_forecast_tensor = torch.tensor(y_forecast, dtype=torch.float32)

    return (
        TensorDataset(X_train_tensor, y_train_tensor),
        TensorDataset(X_test_tensor, y_test_tensor),
        TensorDataset(X_forecast_tensor, y_forecast_tensor),
        scaler,
        dict_df,
        column_label,
        test_max,
        test_min,
        forecast_max,
        forecast_min
    )

