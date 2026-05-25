# text_avg_embedding.py
import os
import re
import numpy as np
import pandas as pd


base_path = "../processed/SDSS/0.1sampling"
csv_name  = "template_param_string_dict_modified.csv"
out_dim   = 32
seed      = 2024
min_count = 1
window    = 5
sg        = 1   # 1: skip-gram, 0: CBOW
# ===========================
def simple_sql_tokenize(sql: str):
    s = str(sql).lower()
    s = re.sub(r"[\n\r\t,()]", " ", s)
    s = re.sub(r"([=<>!]+)", r" \1 ", s)
    s = re.sub(r"(\.)", r" \1 ", s)
    tokens = [t for t in s.split() if t]
    return tokens

def average_vectors(tokens, get_vec, dim):
    if not tokens:
        return np.zeros(dim, dtype=np.float32)
    vecs = []
    for t in tokens:
        v = get_vec(t)
        if v is not None:
            vecs.append(v)
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    arr = np.vstack(vecs)
    return arr.mean(axis=0).astype(np.float32)

def main():
    np.random.seed(seed)
    csv_path = os.path.join(base_path, csv_name)
    out_dir  = base_path
    df = pd.read_csv(csv_path)
    sentences, ids = [], []
    for _, row in df.iterrows():
        tid = row["template_id"]
        sql = str(row["template_sql"])
        toks = simple_sql_tokenize(sql)
        sentences.append(toks)
        ids.append(tid)
    E = int(out_dim)
    use_gensim = False
    try:
        from gensim.models import Word2Vec
        use_gensim = True
    except Exception:
        use_gensim = False

    if use_gensim:
        print("[A3-AVG] Using gensim Word2Vec training...")
        model = Word2Vec(
            sentences=sentences,
            vector_size=E,
            window=window,
            min_count=min_count,
            workers=4,
            sg=sg,
            seed=seed,
        )
        wv = model.wv
        def get_vec(tok: str):
            return wv[tok].astype(np.float32) if tok in wv else None
    else:
        print("[A3-AVG] gensim not available, fallback to stable-hash token vectors...")
        token_cache = {}
        def token2rng(tok: str):
            h = (hash(tok) ^ (seed * 13)) & 0xFFFFFFFF
            return np.random.RandomState(h)
        def get_vec(tok: str):
            if tok in token_cache:
                return token_cache[tok]
            rng = token2rng(tok)
            v = rng.normal(0.0, 1.0, size=E).astype(np.float32)
            token_cache[tok] = v
            return v
    embeddings = np.vstack([average_vectors(toks, get_vec, E) for toks in sentences])
    np.save(os.path.join(out_dir, "template_avgw2v_embeddings.npy"), embeddings)
    out_csv = pd.DataFrame({"template_id": ids})
    out_csv = pd.concat([out_csv, pd.DataFrame(embeddings)], axis=1)
    out_csv.to_csv(os.path.join(out_dir, "template_avgw2v_embeddings.csv"), index=False)
    print(f"[A3-AVG] Saved embeddings: {embeddings.shape} to {out_dir}")

if __name__ == "__main__":
    main()
