# QuadraFormer+: Knowledge-Enhanced Unified Multi-Scale Forecasting for Database Workloads

This repository provides the official implementation of **QuadraFormer+**, the extended journal version of our ICDM 2025 paper *“QuadraFormer: Unified Query and Resource Forecasting for Database Workloads.”*

QuadraFormer+ introduces **knowledge-enhanced SQL modeling (join-tree GNN)**, **scale-adaptive multi-resolution forecasting**, and **robust multi-task training**, enabling unified prediction of SQL query parameters and system resource metrics (e.g., CPU, memory, I/O, QPS) under complex, multi-scale workload dynamics.

---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

- Tested on **Python ≥ 3.9**
- Recommended **PyTorch ≥ 2.0**

---

## 🚀 Quick Start

All configuration options are defined in `run.py` under the `config = { ... }` section.

To run training / evaluation / forecasting:

```bash
python run.py
```

This will execute QuadraFormer+ using the built-in configuration.

---

## ⚙️ Example Config (`run.py`)

```python
# Example usage:
config = {
    "mode": "test",             # [train, test, forecast]
    "interval": 1,              # [None, 1, 6, 12, 24]; QPS: 48
    "data_set": "SDSS",         # [SDSS, tiramisu, tiramisu-sim, alibaba]
    "data_path": "processed\SDSS\0.1sampling",  # Data path [0.1sampling]
    "data_type": "hyper",       # [hyper, sql, resource]

    "model_type": "QuadraFormer_star",
    # Other options:
    # [HistoryMean, QB5000, WFGAN, Stacked_LSTM, Transformer,
    #  PathFormer, QuadraFormer, QuadraFormer_star]
    #  NaiveRepeat, ARIMAModel, RandomForest

    "processed": "True",
    "all_col": "True",
    "less": "Flase",
    "QPS": "Flase",

    "window_size": 32,
    "prediction_length": 32,
    "batch_size": 16,

    "learning_rate": 2e-4,
    "early_stopping": 1e-3,
    "epochs": 10,

    "input_dim": None,
    "output_dim": None,

    "tolerance": 1.5,

    "adaptive_patch": "true",
    "patch_d_min": 1,
    "patch_d_max": 16,

    "embed_variant": "gnn",     # ['gnn', 'rand', 'avgw2v']
}
```

---

## 🧩 Project Structure

| File / Folder                     | Description                                                       |
|-----------------------------------|-------------------------------------------------------------------|
| `Forecaster/model.py`             | Main model definitions                                            |
| `Forecaster/data_loader.py`       | Data loading pipeline                                             |
| `Forecaster/data_process.py`      | Preprocessing scripts                                             |
| `Forecaster/metrics_evaluation.py`| Metric computation                                                |
| `Forecaster/layers/`              | Attention, dilation, GNN, RevIN, experts                         |
| `Forecaster/util/`                | Utilities and time features                                       |
| `Forecaster/saved_models/`        | Model checkpoints                                                 |
| `processed/`                      | Preprocessed dataset directory                                    |
| `run.py`                          | Main script                                                       |
| `requirements.txt`                | Dependencies                                                      |
| `README.md`                       | This file                                                         |

---

## 📊 Datasets

Supported workloads:

- **SDSS Query Log**
- **BusTracker / Tiramisu**
- **BusTracker-sim**
- **Alibaba Cluster Data**

Preprocess with:

```bash
python Forecaster/data_process.py
```

---

## 📄 License

MIT License

```
Copyright (c) 2025
Anonymous Authors
```
