# train_sql_gnn.py
# -*- coding: utf-8 -*-

import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling

# 从你已有的 graph_embedding.py 里导入
from graph_embedding import (
    sql_to_graph,
    build_vocabs,
    nx_to_pyg,
    EdgeAwareSQLGNNEncoder,
)


class LinkPredictor(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_h, edge_index):
        src = edge_index[0]
        dst = edge_index[1]
        h = torch.cat([node_h[src], node_h[dst]], dim=-1)
        return self.mlp(h).squeeze(-1)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_sql_gnn(
    csv_path: str,
    out_dir: str,
    id_col: str = "template_id",
    sql_col: str = "template_sql",
    hidden_dim: int = 64,
    out_dim: int = 32,
    batch_size: int = 4,
    epochs: int = 300,
    lr: float = 1e-3,
    seed: int = 42,
):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    ids = df[id_col].astype(str).tolist()
    sqls = df[sql_col].astype(str).tolist()

    print(f"[Info] Loaded templates: {len(sqls)}")

    graphs = []
    for sql in sqls:
        graphs.append(sql_to_graph(sql))

    vocabs = build_vocabs(graphs)
    vocab_path = os.path.join(out_dir, "template_graph_vocabs.json")
    vocabs.save(vocab_path)
    print(f"[Info] Saved vocab to: {vocab_path}")

    data_list = [nx_to_pyg(G, vocabs) for G in graphs]

    valid_data_list = []
    for d in data_list:
        if d.num_nodes >= 2 and d.edge_index.size(1) > 0:
            valid_data_list.append(d)

    if len(valid_data_list) == 0:
        raise RuntimeError(
            "No valid graph for link-prediction training. "
            "Please check whether SQL graphs contain at least two nodes and edges."
        )

    print(f"[Info] Valid graphs for training: {len(valid_data_list)}")

    loader = DataLoader(valid_data_list, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = EdgeAwareSQLGNNEncoder(
        num_node_tokens=len(vocabs.node_token_vocab),
        num_edge_tokens=len(vocabs.edge_token_vocab),
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        dropout=0.1,
    ).to(device)

    predictor = LinkPredictor(hidden_dim=hidden_dim).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    best_loss = float("inf")
    best_path = os.path.join(out_dir, "trained_sql_gnn.pt")

    for epoch in range(1, epochs + 1):
        encoder.train()
        predictor.train()

        total_loss = 0.0
        total_edges = 0

        for batch in loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            node_h = encoder.encode_nodes(batch)

            pos_edge_index = batch.edge_index

            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=batch.num_nodes,
                num_neg_samples=pos_edge_index.size(1),
                method="sparse",
            )

            pos_logits = predictor(node_h, pos_edge_index)
            neg_logits = predictor(node_h, neg_edge_index)

            logits = torch.cat([pos_logits, neg_logits], dim=0)
            labels = torch.cat([
                torch.ones_like(pos_logits),
                torch.zeros_like(neg_logits),
            ], dim=0)

            loss = F.binary_cross_entropy_with_logits(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(predictor.parameters()),
                max_norm=5.0,
            )
            optimizer.step()

            total_loss += loss.item() * logits.numel()
            total_edges += logits.numel()

        avg_loss = total_loss / max(total_edges, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), best_path)

        if epoch == 1 or epoch % 20 == 0:
            print(f"[Epoch {epoch:04d}] loss={avg_loss:.6f}, best={best_loss:.6f}")

    print(f"[Done] Saved trained SQL GNN checkpoint to: {best_path}")
    print("[Note] Use this checkpoint in graph_embedding.py with --no-random --checkpoint.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=str,
        default="../processed/SDSS/0.1sampling/template_param_string_dict_modified.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="../processed/SDSS/0.1sampling/",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="template_id",
    )
    parser.add_argument(
        "--sql-col",
        type=str,
        default="template_sql",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--out-dim",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train_sql_gnn(
        csv_path=args.csv,
        out_dir=args.out_dir,
        id_col=args.id_col,
        sql_col=args.sql_col,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()