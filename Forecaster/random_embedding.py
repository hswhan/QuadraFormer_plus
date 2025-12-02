# random_embedding.py
import os
import numpy as np
import pandas as pd

base_path = "../processed/SDSS/0.1sampling"
csv_name  = "template_param_string_dict_modified.csv"
out_dim   = 32
seed      = 2024
# ===========================

def main():
    np.random.seed(seed)

    csv_path = os.path.join(base_path, csv_name)
    out_dir  = base_path
    df = pd.read_csv(csv_path)
    ids = df["template_id"].tolist()
    N   = len(ids)
    E   = int(out_dim)
    embeddings = np.random.normal(0.0, 1.0, size=(N, E)).astype(np.float32)
    np.save(os.path.join(out_dir, "template_rand_embeddings.npy"), embeddings)
    out_csv = pd.DataFrame({"template_id": ids})
    out_csv = pd.concat([out_csv, pd.DataFrame(embeddings)], axis=1)
    out_csv.to_csv(os.path.join(out_dir, "template_rand_embeddings.csv"), index=False)
    print(f"[A1-RAND] Saved embeddings: {embeddings.shape} to {out_dir}")

if __name__ == "__main__":
    main()
