import os
import ast
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

def process_output_columns(all_outputs, config, scaler):
    outputs_original = scaler.inverse_transform(
        all_outputs.reshape(-1, all_outputs.shape[-1])
    ).reshape(all_outputs.shape)
    cols = outputs_original.shape[2]
    id_col = outputs_original[:, :, 0:1]
    id_col = np.round(id_col)
    time_cols = outputs_original[:, :, 1:7]
    time_cols = np.round(time_cols)
    param_start = 13
    param_end = cols - 6
    param_cols = outputs_original[:, :, param_start:param_end]
    param_half = param_cols.shape[2] // 2
    param_selected = param_cols[:, :, :param_half]
    resource_cols = outputs_original[:, :, -6:]
    resource_selected = resource_cols[:, :, :3]
    combined_data = np.concatenate([id_col, time_cols, param_selected, resource_selected], axis=2)
    save_dir = os.path.join("rst", f"window_{config['window_size']}_{config['prediction_length']}")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "outputs.npy"), combined_data)
    print(f"Outputs results saved in：{save_dir}")


def match_range(predicted_col, ground_truth_col, condition, data_max, data_min, string_dict):
    cnt_diffs = []
    matches = []
    if condition == "between":
        predicted_left = predicted_col[:, 0]
        predicted_right = predicted_col[:, 1]
        gt_left = ground_truth_col[:, 0]
        gt_right = ground_truth_col[:, 1]
        predicted_min = np.min(predicted_left)
        predicted_max = np.max(predicted_right)
        gt_min = np.min(gt_left)
        gt_max = np.max(gt_right)
        denom = np.abs(gt_max - gt_min)
        if denom != 0:
            extra = np.abs((predicted_max - predicted_min) - (gt_max - gt_min))
            cnt_diff = extra / denom
        else:
            cnt_diff = 0.0
        cnt_diff = np.nan_to_num(cnt_diff, nan=0.0, posinf=0.0, neginf=0.0)
        cnt_diffs = [cnt_diff] * len(predicted_col)
        left_matches = [1 if predicted_left[i] <= gt_left[i] else 0 for i in range(len(predicted_col))]
        right_matches = [1 if predicted_right[i] >= gt_right[i] else 0 for i in range(len(predicted_col))]
        matches.append((left_matches, right_matches))
    elif condition == "in" or condition == "not in":
        predicted_set = set(predicted_col)
        ground_truth_set = set(ground_truth_col)
        set_diff = predicted_set - ground_truth_set
        cnt_diff = len(set_diff) / len(ground_truth_set) if len(ground_truth_set) > 0 else 0
        for i in range(len(predicted_col)):
            matches.append(1 if predicted_col[i] in ground_truth_set else 0)
            cnt_diffs.append(cnt_diff)
    elif condition == "like":
        predicted_set = set(predicted_col)
        ground_truth_set = set(ground_truth_col)
        set_diff = predicted_set - ground_truth_set
        cnt_diff = len(set_diff) / len(ground_truth_set) if len(ground_truth_set) > 0 else 0
        for i in range(len(predicted_col)):
            matches.append(1 if predicted_col[i] in ground_truth_set else 0)
            cnt_diffs.append(cnt_diff)
    elif condition == "<" or condition == "<=":
        matches = [1 if predicted_col[i] >= ground_truth_col[i] else 0 for i in range(len(predicted_col))]
        pred_max = np.max(predicted_col)
        gt_max = np.max(ground_truth_col)
        real_range = gt_max - data_min
        pred_range = pred_max - data_min
        cnt_diff = 0.0 if real_range <= 0 else abs(pred_range - real_range) / real_range
        cnt_diffs = [cnt_diff] * len(predicted_col)
    elif condition == ">" or condition == ">=":
        matches = [1 if predicted_col[i] <= ground_truth_col[i] else 0 for i in range(len(predicted_col))]
        pred_min = np.min(predicted_col)
        gt_min = np.min(ground_truth_col)
        real_range = data_max - gt_min
        pred_range = data_max - pred_min
        cnt_diff = 0.0 if real_range <= 0 else abs(pred_range - real_range) / real_range
        cnt_diffs = [cnt_diff] * len(predicted_col)
    return matches, cnt_diffs


def calculate_range_metrics(predicted, ground_truth, conditions, data_max, data_min, string_dict):
    all_matches = []
    all_cnt_diffs = []
    col = 0
    while col < predicted.shape[1]:
        condition = conditions[col]
        predicted_col = predicted[:, col]
        ground_truth_col = ground_truth[:, col]
        if condition == "between" and col + 1 < predicted.shape[1] and conditions[col + 1] == "between":
            next_predicted_col = predicted[:, col + 1]
            next_ground_truth_col = ground_truth[:, col + 1]
            merged_predicted_col = np.column_stack((predicted_col, next_predicted_col))
            merged_ground_truth_col = np.column_stack((ground_truth_col, next_ground_truth_col))
            matches, cnt_diffs = match_range(merged_predicted_col, merged_ground_truth_col, "between",
                                             data_max[col], data_min[col], string_dict)
            all_matches.append(matches[0][0])
            all_matches.append(matches[0][1])
            all_cnt_diffs.append(cnt_diffs)
            all_cnt_diffs.append(cnt_diffs)
            col += 2
        else:
            matches, cnt_diffs = match_range(predicted_col, ground_truth_col, condition,
                                             data_max[col], data_min[col], string_dict)
            all_matches.append(matches)
            all_cnt_diffs.append(cnt_diffs)
            col += 1
    return all_matches, all_cnt_diffs


def preprocess_numeric(ground_truth, predicted, column_labels):
    D = ground_truth.shape[1]
    for i in range(D):
        data_type = str(column_labels[i, 2]).lower()
        if data_type in ["int", "string"]:
            ground_truth[:, i] = np.rint(ground_truth[:, i])
            predicted[:, i] = np.rint(predicted[:, i])
        elif data_type == "float":
            ground_truth[:, i] = np.round(ground_truth[:, i], 2)
            predicted[:, i] = np.round(predicted[:, i], 2)
    return ground_truth, predicted


def match_query_using_range(gt_row, pred_row, dict_df, column_labels, exact_match_index, range_match_index, config):
    for idx in exact_match_index:
        if column_labels[idx][2] in ['int', 'float']:
            tolerance = 1.5 * abs(gt_row[idx])
        else:
            if config['model_type'] in ['PathFormer', 'QuadraFormer', 'QuadraFormer_woc', 'QuadraFormer_woa', 'QuadraFormer_wos']:
                tolerance = 1.5 * abs(gt_row[idx])
            else:
                tolerance = 0.99 * abs(gt_row[idx])
        if abs(gt_row[idx] - pred_row[idx]) > tolerance:
            return False
    col = 0
    while col < len(range_match_index):
        idx = range_match_index[col]
        condition = str(column_labels[idx, 1]).lower()
        if condition == "between" and col + 1 < len(range_match_index):
            idx_next = range_match_index[col + 1]
            gt_col = np.array([[gt_row[idx], gt_row[idx_next]]])
            pred_col = np.array([[pred_row[idx], pred_row[idx_next]]])
            matches, _ = match_range(pred_col, gt_col, "between", 0, 0, column_labels[:, 3])
            left_match = matches[0][0][0]
            right_match = matches[0][1][0]
            if left_match != 1 or right_match != 1:
                return False
            col += 2
        else:
            gt_col = np.array([gt_row[idx]])
            pred_col = np.array([pred_row[idx]])
            matches, _ = match_range(pred_col, gt_col, condition, 0, 0, column_labels[:, 3])
            if matches[0] != 1:
                return False
            col += 1
    return True


def greedy_bipartite_matching(ground_truth, predicted, dict_df, column_labels, exact_match_index, range_match_index, config):
    N = ground_truth.shape[0]
    M = predicted.shape[0]
    matched_pred = [False] * M
    matching = []
    for i in range(N):
        gt_row = ground_truth[i, :]
        template_id = int(gt_row[0])
        param_columns = dict_df[template_id][4]
        param_columns = list(map(int, ast.literal_eval(param_columns)))
        # param_columns = [int(x) + 31 for x in ast.literal_eval(param_columns)]#新加了31列sql文本特征
        exact_match_index_new = [idx for idx in exact_match_index if idx in param_columns]
        range_match_index_new = [idx for idx in range_match_index if idx in param_columns]
        local_exact_match_index = [param_columns.index(idx) for idx in exact_match_index_new]
        local_range_match_index = [param_columns.index(idx) for idx in range_match_index_new]
        gt_sub = gt_row[param_columns]
        col_sub = column_labels[param_columns]
        for j in range(M):
            if not matched_pred[j]:
                pred_row = predicted[j, :]
                pred_sub = pred_row[param_columns]
                if match_query_using_range(gt_sub, pred_sub, dict_df, col_sub,
                                           local_exact_match_index, local_range_match_index, config):
                    matching.append((i, j))
                    matched_pred[j] = True
                    break
    return matching


def compute_f1_metrics(matching, ground_truth, predicted):
    recall = len(matching) / len(ground_truth) if len(ground_truth) > 0 else 0
    precision = len(matching) / len(predicted) if len(predicted) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return recall, precision, f1


def _collect_indices_by_name(all_columns):
    name2idx = {c: i for i, c in enumerate(all_columns)}
    # embed
    embed_idx = [i for i, c in enumerate(all_columns) if c.startswith("embed_")]
    # timestamp
    ts_main_names = ["YY", "MM", "DD", "HOUR", "MIN", "SEC"]
    ts_val_names = [f"{n}_val" for n in ts_main_names]
    ts_main_idx = [name2idx[n] for n in ts_main_names if n in name2idx]
    ts_val_idx = [name2idx[n] for n in ts_val_names if n in name2idx]
    # feature
    pat_main = re.compile(r"^feature_(\d+)$")
    pat_val = re.compile(r"^feature_(\d+)_val$")
    feature_main_idx = []
    feature_val_idx = []
    for i, c in enumerate(all_columns):
        if pat_main.match(c):
            feature_main_idx.append(i)
        elif pat_val.match(c):
            feature_val_idx.append(i)
    # resource
    resource_names = ["elapsed", "busy", "rows", "elapsed_val", "busy_val", "rows_val"]
    resource_idx = [name2idx[n] for n in resource_names if n in name2idx]
    if len(resource_idx) != 6:
        D = len(all_columns)
        resource_idx = list(range(D - 6, D))
    template_id_idx = name2idx.get("template_id", None)

    return (name2idx, embed_idx, ts_main_idx, ts_val_idx,
            feature_main_idx, feature_val_idx, resource_idx, template_id_idx)

def _remap_column_labels_to_feature_main(column_labels, feature_main_idx, all_columns):
    D_sql = len(feature_main_idx)
    # 默认：between/float
    column_labels_sql = np.empty((D_sql, 4), dtype=object)
    for j in range(D_sql):
        column_labels_sql[j, 0] = j
        column_labels_sql[j, 1] = "between"  # 默认回归用 between
        column_labels_sql[j, 2] = "float"
        column_labels_sql[j, 3] = "nan"
    if isinstance(column_labels, np.ndarray):
        df = pd.DataFrame(column_labels, columns=["col_name", "match_type", "value_type", "string_dict"])
    else:
        df = column_labels.copy()
        ren = {}
        for a in ["col_name", "Column", "column", "name", "ColName"]:
            if a in df.columns: ren[a] = "col_name"; break
        for a in ["match_type", "Label", "MatchType", "match", "type"]:
            if a in df.columns: ren[a] = "match_type"; break
        for a in ["value_type", "DataType", "ValueType", "dtype"]:
            if a in df.columns: ren[a] = "value_type"; break
        for a in ["string_dict", "StringDict", "string", "dict"]:
            if a in df.columns: ren[a] = "string_dict"; break
        if ren:
            df = df.rename(columns=ren)
    exact_local = set()
    range_local = set()
    def to_local_idx(v):
        if isinstance(v, (int, np.integer)):
            k = int(v)
        elif isinstance(v, str) and v.strip().isdigit():
            k = int(v.strip())
        else:
            return None
        if 1 <= k <= D_sql:
            return k - 1
        if 0 <= k < D_sql:
            return k
        return None

    for _, row in df.iterrows():
        j = to_local_idx(row["col_name"])
        if j is None:
            continue
        vt = str(row.get("value_type", "float")).lower()
        mt = str(row.get("match_type", "") or "").lower()
        sd = row.get("string_dict", "nan")

        if mt == "":
            mt = "between" if vt == "float" else "exact_match"

        column_labels_sql[j, 1] = mt
        column_labels_sql[j, 2] = vt
        column_labels_sql[j, 3] = sd

        if mt == "exact_match":
            exact_local.add(j)
        else:
            range_local.add(j)

    exact_local_idx = sorted(exact_local)
    range_local_idx = sorted(range_local)
    return column_labels_sql, exact_local_idx, range_local_idx

def calculate_metrics(outputs, batch_y, config, dict_df, column_labels, scaler, data_max, data_min):
    import os
    import numpy as np
    import pandas as pd
    import torch
    from datetime import timedelta
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    metrics = {}
    output_path = os.path.join(
        'rst',
        f'window_{config["window_size"]}_{config["prediction_length"]}',
        f'{config["model_type"]}'
    )
    os.makedirs(output_path, exist_ok=True)
    mode = str(config.get('mode', 'train')).lower()
    data_type = str(config.get('data_type', 'sql')).lower()
    outputs = outputs.detach().cpu().numpy() if isinstance(outputs, torch.Tensor) else outputs
    batch_y = batch_y.detach().cpu().numpy() if isinstance(batch_y, torch.Tensor) else batch_y

    if scaler is not None:
        outputs = scaler.inverse_transform(outputs.reshape(-1, outputs.shape[-1])).reshape(outputs.shape)
        batch_y = scaler.inverse_transform(batch_y.reshape(-1, batch_y.shape[-1])).reshape(batch_y.shape)
    outputs = np.clip(outputs, data_min, data_max)
    batch_y = np.clip(batch_y, data_min, data_max)


    if mode == 'forecast':
        np.save(os.path.join(output_path, "forecast_outputs.npy"), outputs)
        np.save(os.path.join(output_path, "forecast_batch_y.npy"), batch_y)
        print(f"Forecast result have been saved at: {os.path.join(output_path, 'forecast_outputs.npy')}")
    else:
        np.save(os.path.join(output_path, f"{mode}_outputs.npy"), outputs)
        np.save(os.path.join(output_path, f"{mode}_batch_y.npy"), batch_y)
    B, L, D = outputs.shape

    all_columns = config.get('all_columns', [f"col_{i}" for i in range(D)])
    name2idx = {c: i for i, c in enumerate(all_columns)}

    def _has_named_schema(cols):
        return (
            any(c.startswith("embed_") for c in cols) or
            any(n in cols for n in ["YY", "MM", "DD", "HOUR", "MIN", "SEC"]) or
            any(n in cols for n in ["YY_val", "MM_val", "DD_val", "HOUR_val", "MIN_val", "SEC_val"]) or
            any(c.startswith("feature_") for c in cols) or
            ("template_id" in cols)
        )

    named_schema = _has_named_schema(all_columns)

    ts_main_idx, ts_val_idx = [], []
    feature_main_idx = []
    template_id_idx = name2idx.get("template_id", None)

    if named_schema:
        embed_idx = [i for i, c in enumerate(all_columns) if c.startswith("embed_")]
        ts_main_names = ["YY", "MM", "DD", "HOUR", "MIN", "SEC"]
        ts_val_names  = [f"{n}_val" for n in ts_main_names]
        ts_main_idx = [name2idx[n] for n in ts_main_names if n in name2idx]
        ts_val_idx  = [name2idx[n] for n in ts_val_names if n in name2idx]
        tid_cnt = 1 if (template_id_idx is not None) else 0

        if data_type in ("sql", "hyper"):
            param_start = len(embed_idx) + len(ts_main_idx) + len(ts_val_idx) + tid_cnt
            feature_main_idx = [
                i for i, c in enumerate(all_columns)
                if (i >= param_start) and c.startswith("feature_") and (not c.endswith("_val"))
            ]
        else:
            param_start = len(ts_main_idx) + len(ts_val_idx) + tid_cnt
            feature_main_idx = []
    else:
        ts_main_idx = list(range(1, min(7, D)))   # 1..6
        ts_val_idx  = list(range(7, min(13, D)))  # 7..12

        if data_type in ("sql", "hyper"):
            param_start = 45 if D > 45 else max(0, min(45, D))
            feature_main_idx = [i for i in range(param_start, D) if not str(all_columns[i]).endswith("_val")]
            if not feature_main_idx:
                feature_main_idx = list(range(param_start, D))
        else:
            param_start = 13 if D > 13 else max(0, min(13, D))
            feature_main_idx = []

    first_param = all_columns[param_start] if param_start < D else "OUT_OF_RANGE"
    print(f"[INFO] param_start={param_start}, first param col={first_param}")
    print(f"[Cols] timestamps(main+val) = {[all_columns[i] for i in (ts_main_idx + ts_val_idx)]}")
    if data_type in ("sql", "hyper"):
        print(f"[Cols] sql (head) = {[all_columns[i] for i in feature_main_idx[:10]]}")
    else:
        print(f"[Cols] sql (head) = []  (resource 模式不评 SQL)")

    if data_type in ("sql", "hyper") and feature_main_idx:
        sql_outputs = outputs[:, :, feature_main_idx]
        sql_labels  = batch_y[:, :, feature_main_idx]
        sql_datamax = data_max[feature_main_idx]
        sql_datamin = data_min[feature_main_idx]
        D_sql = sql_outputs.shape[2]
    else:
        sql_outputs = sql_labels = None
        sql_datamax = sql_datamin = None
        D_sql = 0

    timestamp_index   = ts_main_idx
    timestamp_outputs = outputs[:, :, timestamp_index] if timestamp_index else None
    timestamp_labels  = batch_y[:, :, timestamp_index] if timestamp_index else None

    if data_type == "resource":
        resource_start = param_start
        resource_end = D
        half_len = max(0, (resource_end - resource_start) // 2)
        resource_index = list(range(resource_start, resource_start + half_len))
        print(f"[Cols] resource (slice) head = {[all_columns[i] for i in resource_index[:6]]}")
        resource_outputs = outputs[:, :, resource_index]
        resource_labels  = batch_y[:, :, resource_index]
    else:
        resource_outputs = resource_labels = None

    if sql_outputs is not None:
        column_labels_sql, exact_match_index, range_match_index = _remap_column_labels_to_feature_main(
            column_labels, feature_main_idx, all_columns
        )

        flat_sql_outputs = sql_outputs.reshape(B * L, D_sql)
        flat_sql_labels  = sql_labels.reshape(B * L, D_sql)

        flat_sql_labels, flat_sql_outputs = preprocess_numeric(
            flat_sql_labels, flat_sql_outputs, column_labels_sql
        )
        have_template_col = (template_id_idx is not None)

        # —— Forecast: NEXT-T —— #
        if mode == 'forecast' and config['interval'] != 'None':
            # 仅对 sql/hyper 评 F1
            if data_type not in ('sql', 'hyper'):
                metrics = {"Forecast NEXT-T Metrics": {"Recall": 0.0, "Precision": 0.0, "F1": 0.0}}
                return metrics

            col_arr = np.asarray(column_labels.values if hasattr(column_labels, "values") else column_labels)
            if col_arr.ndim != 2 or col_arr.shape[1] != 4:
                raise ValueError(f"column_labels shape must be (N,4), got {col_arr.shape}")
            half_len_from_labels = int(col_arr.shape[0])
            sql_range_start = 13
            sql_range_end   = D - 6 if data_type == 'hyper' else D
            max_len_by_data = max(0, sql_range_end - sql_range_start)
            use_len = min(half_len_from_labels, max_len_by_data)
            sql_index_old = [0] + list(range(sql_range_start, sql_range_start + use_len))

            flat_labels_fore  = batch_y.reshape(B * L, D)[:, sql_index_old]
            flat_outputs_fore = outputs.reshape(B * L, D)[:, sql_index_old]

            tid_row = np.array([0, 'exact_match', 'int', 'nan'], dtype=object)
            column_labels_fore = np.vstack([tid_row, col_arr[:use_len, :]])

            flat_labels_fore, flat_outputs_fore = preprocess_numeric(
                flat_labels_fore, flat_outputs_fore, column_labels_fore
            )

            exact_idx, range_idx = [], []
            for r in column_labels_fore:
                if r[1] == 'exact_match':
                    exact_idx.append(r[0])
                else:
                    range_idx.append(r[0])

            t_idx = list(range(1, 6 + 1))  # 1..6 -> [YY,MM,DD,HOUR,MIN,SEC]
            labels_2d  = np.rint(batch_y[:, :, t_idx]).astype(int).reshape(-1, 6)
            outputs_2d = np.rint(outputs[:,  :, t_idx]).astype(int).reshape(-1, 6)

            df_labels  = pd.DataFrame(labels_2d,  columns=["year","month","day","hour","minute","second"])
            df_outputs = pd.DataFrame(outputs_2d, columns=["year","month","day","hour","minute","second"])

            dl = pd.to_datetime(df_labels[['year','month','day','hour','minute','second']],  errors='coerce')
            do = pd.to_datetime(df_outputs[['year','month','day','hour','minute','second']], errors='coerce')

            valid_mask = dl.notna() & do.notna()
            valid_mask = valid_mask.values  # ndarray[*,1]
            valid_mask = valid_mask.reshape(-1)

            n_all = len(df_labels)
            n_valid = int(valid_mask.sum())

            ds = str(config.get('data_set', '')).lower()
            time_axis = dl if ('tiramisu' in ds) else do

            def window_on(series_time, hours, frac_fallback=0.33):
                series = series_time.copy()
                series_valid = series[valid_mask]
                if len(series_valid) == 0:
                    k = max(1, int(n_all * frac_fallback))
                    s = max(0, (n_all - k) // 2)
                    return list(range(s, s + k))

                series_floor = pd.to_datetime(series_valid).dt.floor('s')
                min_time = series_floor.min()
                max_time = series_floor.max()
                interval = timedelta(hours=int(hours))

                if pd.isna(min_time) or pd.isna(max_time):
                    k = max(1, int(n_all * frac_fallback))
                    s = max(0, (n_all - k) // 2)
                    return list(range(s, s + k))

                total_duration = max_time - min_time
                if total_duration <= interval:
                    idx_valid = series_valid.index.to_numpy()
                else:
                    median_time = series_floor.median()
                    start_time = (median_time - interval / 2).floor('s')
                    end_time   = (median_time + interval / 2).floor('s')
                    idx_valid = series_floor[
                        (series_floor >= start_time) & (series_floor <= end_time)
                        ].index.to_numpy()

                idx = idx_valid.tolist()

                if 'tiramisu' in ds:
                    if len(idx) == 0 or len(idx) >= int(0.9 * n_all):
                        k = max(1, int(n_all * 0.33))
                        s = max(0, (n_all - k) // 2)
                        idx = list(range(s, s + k))
                else:
                    if len(idx) == 0:
                        k = max(1, n_all // 2)
                        s = max(0, (n_all - k) // 2)
                        idx = list(range(s, s + k))
                return idx

            indices = window_on(time_axis, config['interval'])

            if len(indices) == 0:
                k = max(1, int(n_all * (0.33 if 'tiramisu' in ds else 0.5)))
                s = max(0, (n_all - k) // 2)
                indices = list(range(s, s + k))

            predicted_workload = flat_outputs_fore[indices]
            true_workload      = flat_labels_fore

            assert predicted_workload.shape[1] == true_workload.shape[1] == column_labels_fore.shape[0], \
                f"shape mismatch: pred {predicted_workload.shape}, true {true_workload.shape}, labels {column_labels_fore.shape}"

            matching = greedy_bipartite_matching(
                true_workload, predicted_workload,
                dict_df, column_labels_fore,
                exact_idx, range_idx, config
            )
            recall, precision, f1 = compute_f1_metrics(matching, true_workload, predicted_workload)

            print(f"True_workload size: {len(true_workload)}, Predicated_workload size: {len(predicted_workload)}")
            print(f"NEXT-T Recall : {recall}")
            print(f"NEXT-T Precision: {precision}")
            print(f"NEXT-T F1: {f1}")

            metrics = {"Forecast NEXT-T Metrics": {"Recall": recall, "Precision": precision, "F1": f1}}



        if mode in ['train', 'test']:
            final_feature_matches = {}
            final_cnt_diff_matches = {}

            sub_labels  = flat_sql_labels
            sub_outputs = flat_sql_outputs

            exact_cols = list(exact_match_index)
            range_cols = list(range_match_index)

            # exact
            if exact_cols:
                exact_match_result = np.equal(sub_outputs[:, exact_cols],
                                              sub_labels[:, exact_cols]).astype(int)
            else:
                exact_match_result = np.empty((sub_labels.shape[0], 0), dtype=int)

            # range
            if range_cols:
                sub_range_labels  = sub_labels[:, range_cols]
                sub_range_outputs = sub_outputs[:, range_cols]

                conditions   = np.array([column_labels_sql[j, 1] for j in range_cols])
                datamax_arr  = np.array([sql_datamax[j] for j in range_cols])
                datamin_arr  = np.array([sql_datamin[j] for j in range_cols])
                string_dict_arr = np.array([column_labels_sql[j, 3] for j in range_cols])

                range_match_predictions, cnt_diff = calculate_range_metrics(
                    sub_range_outputs, sub_range_labels, conditions, datamax_arr, datamin_arr, string_dict_arr
                )
                range_match_result = np.array(range_match_predictions).T  # [N, len(range_cols)]
                cnt_diff_result    = np.array(cnt_diff).T
            else:
                range_match_result = np.empty((sub_labels.shape[0], 0), dtype=int)
                cnt_diff_result    = np.empty((sub_labels.shape[0], 0), dtype=float)

            for k, j in enumerate(exact_cols):
                final_feature_matches.setdefault(j, []).append(exact_match_result[:, k])

            for k, j in enumerate(range_cols):
                final_feature_matches.setdefault(j, []).append(range_match_result[:, k])
                final_cnt_diff_matches.setdefault(j, []).append(cnt_diff_result[:, k])

            total_match_list = []
            total_cnt_diff_list = []
            all_cols_local = sorted(final_feature_matches.keys())
            for j in all_cols_local:
                aggregated = np.concatenate(final_feature_matches[j], axis=0)
                total_match_list.append(aggregated)
                if j in final_cnt_diff_matches:
                    aggregated_cnt = np.concatenate(final_cnt_diff_matches[j], axis=0)
                    total_cnt_diff_list.append(aggregated_cnt)
                else:
                    total_cnt_diff_list.append(np.array([]))

            accuracies = np.array([np.mean(arr) for arr in total_match_list if arr.size > 0])
            cnt_diffs  = np.array([np.mean(arr) for arr in total_cnt_diff_list if arr.size > 0])
            mean_accuracy = float(np.mean(accuracies)) if accuracies.size > 0 else None
            mean_cnt_diff = float(np.mean(cnt_diffs))  if cnt_diffs.size  > 0 else None
            print(f"All Parameter Mean Accuracy: {mean_accuracy}")
            print(f"Range Parameter Mean cnt-diff: {mean_cnt_diff}")

            df = pd.concat([pd.Series(accuracies), pd.Series(cnt_diffs)], axis=1)
            df.columns = ["accuracy", "cnt_diff"]
            df.to_csv(os.path.join(output_path, "evaluation_metrics.csv"), index=False)

            metrics["SQL Metrics"] = {"ACC": mean_accuracy, "CNT-DIFF": mean_cnt_diff}

    if data_type == "hyper" and mode != 'forecast':
        wanted6 = ["elapsed", "busy", "rows", "elapsed_val", "busy_val", "rows_val"]
        name2idx = {c: i for i, c in enumerate(all_columns)}
        if all(n in name2idx for n in wanted6):
            res6 = [name2idx[n] for n in wanted6]
        else:
            start6 = max(0, D - 6)
            res6 = list(range(start6, D))
        resource_index_h = res6[:3]

        if resource_index_h:
            print(f"[Cols] hyper-resource (slice) head = {[all_columns[i] for i in resource_index_h]}")
            resource_outputs_h = outputs[:, :, resource_index_h]
            resource_labels_h = batch_y[:, :, resource_index_h]

            mse_h = mean_squared_error(resource_labels_h.flatten(), resource_outputs_h.flatten())
            mae_h = mean_absolute_error(resource_labels_h.flatten(), resource_outputs_h.flatten())
            print(f"[Hyper] MSE: {mse_h}")
            print(f"[Hyper] MAE: {mae_h}")
            metrics["Hyper Resource Metrics"] = {"MSE": mse_h, "MAE": mae_h}
        else:
            print("[Hyper] resource slice is empty; skip MSE/MAE.")

    if resource_outputs is not None and mode != 'forecast':
        mse = mean_squared_error(resource_labels.flatten(), resource_outputs.flatten())
        mae = mean_absolute_error(resource_labels.flatten(), resource_outputs.flatten())
        print(f"MSE: {mse}")
        print(f"MAE: {mae}")
        metrics["Resource Metrics"] = {"MSE": mse, "MAE": mae}

    return metrics
