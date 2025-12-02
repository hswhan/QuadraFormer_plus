# graph_embedding.py
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import networkx as nx
import sqlparse

########################################
# 1. SQL -> NetworkX Graph
########################################
def sql_to_graph(sql: str):
    parsed = sqlparse.parse(sql)[0]
    tokens = [t for t in parsed.tokens if not t.is_whitespace]

    G = nx.Graph()
    for t in tokens:
        if t.ttype is None:
            val = t.value.lower()
            if val.startswith("from") or val.startswith("join"):
                words = [w.strip() for w in val.replace("\n", " ").split(" ") if w.strip()]
                for w in words:
                    if "." in w and not w.startswith("select"):
                        G.add_node(w, type="table")
            if val.startswith("where"):
                predicates = ["=", "<", ">", "<=", ">=", "like", "between", "in", "not in"]
                words = val.replace("\n", " ").replace("  ", " ").split(" ")
                for p in predicates:
                    if p in words:
                        G.add_node(p, type="predicate")
    return G

########################################
# 2. NetworkX -> PyG Data
########################################
def nx_to_pyg(G: nx.Graph):
    mapping = {node: i for i, node in enumerate(G.nodes)}
    edge_index = []
    for u, v in G.edges:
        edge_index.append([mapping[u], mapping[v]])
    if len(edge_index) == 0:
        edge_index = [[0, 0]]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    x = []
    for node, attr in G.nodes(data=True):
        if attr.get("type") == "table":
            x.append([1, 0])
        else:
            x.append([0, 1])
    if len(x) == 0:
        x = [[1, 0]]
    x = torch.tensor(x, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

########################################
# 3. GNN Encoder
########################################
class SQLGNNEncoder(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=32, out_dim=32):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)

########################################
# 4. Maibn Process
########################################
def main():
    base_path = "../processed/tiramisu-sim/"
    csv_path = os.path.join(base_path, "template_param_string_dict_modified.csv")
    out_dir = base_path

    df = pd.read_csv(csv_path)
    graphs, ids = [], []
    for _, row in df.iterrows():
        template_id = row["template_id"]
        sql = str(row["template_sql"])
        G = sql_to_graph(sql)
        data = nx_to_pyg(G)
        data.y = torch.tensor([0])  # dummy label
        graphs.append(data)
        ids.append(template_id)

    loader = DataLoader(graphs, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SQLGNNEncoder(out_dim=32).to(device)

    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = model(batch)
            embeddings.append(emb.cpu().numpy())
    embeddings = np.vstack(embeddings)

    np.save(os.path.join(out_dir, "template_gnn_embeddings.npy"), embeddings)
    out_csv = pd.DataFrame({"template_id": ids})
    out_csv = pd.concat([out_csv, pd.DataFrame(embeddings)], axis=1)
    out_csv.to_csv(os.path.join(out_dir, "template_gnn_embeddings.csv"), index=False)
    print(f"Saved embeddings: {embeddings.shape} to {out_dir}")


if __name__ == "__main__":
    main()

