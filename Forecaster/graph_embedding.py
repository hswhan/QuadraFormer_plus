# graph_embedding_fixed.py
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import networkx as nx

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool

try:
    import sqlglot
    from sqlglot import exp
    HAS_SQLGLOT = True
except Exception:
    HAS_SQLGLOT = False


# ============================================================
# 1. Type definitions
# ============================================================

NODE_TYPES = {
    "TABLE": 0,
    "COLUMN": 1,
    "PREDICATE": 2,
    "UNKNOWN": 3,
}

EDGE_TYPES = {
    "SELF": 0,
    "TABLE_COLUMN": 1,
    "JOIN_TABLE": 2,
    "JOIN_COLUMN": 3,
    "FILTER_EQ": 4,
    "FILTER_RANGE": 5,
    "FILTER_LIKE": 6,
    "FILTER_IN": 7,
    "FILTER_OTHER": 8,
}


COMPARISON_CLASSES = ()
if HAS_SQLGLOT:
    COMPARISON_CLASSES = (
        exp.EQ,
        exp.GT,
        exp.GTE,
        exp.LT,
        exp.LTE,
        exp.Like,
        exp.In,
        exp.Between,
        exp.NEQ,
    )


# ============================================================
# 2. Vocabulary
# ============================================================

class Vocab:
    def __init__(self):
        self.token_to_id: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.id_to_token: List[str] = ["<PAD>", "<UNK>"]

    def add(self, token: str) -> int:
        token = self.normalize(token)
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.id_to_token)
            self.id_to_token.append(token)
        return self.token_to_id[token]

    def get(self, token: str) -> int:
        token = self.normalize(token)
        return self.token_to_id.get(token, self.token_to_id["<UNK>"])

    @staticmethod
    def normalize(token: str) -> str:
        if token is None:
            return "<UNK>"
        return str(token).strip().lower()

    def to_dict(self):
        return {
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
        }

    @classmethod
    def from_dict(cls, d):
        v = cls()
        v.token_to_id = d["token_to_id"]
        v.id_to_token = d["id_to_token"]
        return v

    def __len__(self):
        return len(self.id_to_token)


@dataclass
class GraphVocabs:
    node_token_vocab: Vocab
    edge_token_vocab: Vocab

    def save(self, path: str):
        obj = {
            "node_token_vocab": self.node_token_vocab.to_dict(),
            "edge_token_vocab": self.edge_token_vocab.to_dict(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls(
            node_token_vocab=Vocab.from_dict(obj["node_token_vocab"]),
            edge_token_vocab=Vocab.from_dict(obj["edge_token_vocab"]),
        )


# ============================================================
# 3. Graph construction helpers
# ============================================================

def normalize_sql(sql: str) -> str:
    sql = str(sql)
    sql = sql.replace("\n", " ")
    sql = re.sub(r"\s+", " ", sql)
    return sql.strip()


def add_node(
    G: nx.MultiDiGraph,
    node_id: str,
    node_type: str,
    token: str,
    label: Optional[str] = None,
    **kwargs,
):
    if node_id not in G:
        G.add_node(
            node_id,
            node_type=node_type,
            token=token,
            label=label or token,
            **kwargs,
        )


def add_bidirectional_edge(
    G: nx.MultiDiGraph,
    u: str,
    v: str,
    edge_type: str,
    edge_token: str,
    **kwargs,
):
    if u not in G or v not in G:
        return

    G.add_edge(
        u,
        v,
        edge_type=edge_type,
        edge_token=edge_token,
        **kwargs,
    )
    G.add_edge(
        v,
        u,
        edge_type=edge_type,
        edge_token=edge_token,
        **kwargs,
    )


def add_self_loops(G: nx.MultiDiGraph):
    for n in list(G.nodes):
        G.add_edge(
            n,
            n,
            edge_type="SELF",
            edge_token="SELF",
        )


def table_node_id(alias: str) -> str:
    return f"T::{alias.lower()}"


def column_node_id(alias: str, col: str) -> str:
    return f"C::{alias.lower()}.{col.lower()}"


def predicate_node_id(alias: str, col: str, op: str, raw: str) -> str:
    h = abs(hash(raw)) % (10 ** 12)
    return f"P::{alias.lower()}.{col.lower()}::{op.lower()}::{h}"


def ensure_table_node(
    G: nx.MultiDiGraph,
    alias: str,
    table_name: str,
):
    alias = alias.lower()
    table_name = table_name.lower()
    nid = table_node_id(alias)
    add_node(
        G,
        nid,
        node_type="TABLE",
        token=f"table::{table_name}",
        label=alias,
        alias=alias,
        table_name=table_name,
    )
    return nid


def ensure_column_node(
    G: nx.MultiDiGraph,
    alias: str,
    col: str,
    table_name: Optional[str] = None,
):
    alias = alias.lower()
    col = col.lower()
    table_nid = table_node_id(alias)
    col_nid = column_node_id(alias, col)

    add_node(
        G,
        col_nid,
        node_type="COLUMN",
        token=f"column::{table_name or alias}.{col}",
        label=f"{alias}.{col}",
        alias=alias,
        column=col,
        table_name=table_name or alias,
    )

    if table_nid in G:
        add_bidirectional_edge(
            G,
            table_nid,
            col_nid,
            edge_type="TABLE_COLUMN",
            edge_token=f"has_column::{col}",
        )

    return col_nid


def comparison_to_edge_type(op: str) -> str:
    op = op.lower()
    if op in {"=", "eq"}:
        return "FILTER_EQ"
    if op in {">", ">=", "<", "<=", "between", "neq", "!="}:
        return "FILTER_RANGE"
    if op == "like":
        return "FILTER_LIKE"
    if op == "in":
        return "FILTER_IN"
    return "FILTER_OTHER"


# ============================================================
# 4. SQLGlot parser
# ============================================================

def table_full_name(t) -> str:
    parts = []
    db = t.args.get("db")
    catalog = t.args.get("catalog")

    if catalog is not None:
        parts.append(str(catalog).lower())
    if db is not None:
        parts.append(str(db).lower())

    name = getattr(t, "name", None)
    if name:
        parts.append(str(name).lower())

    if not parts:
        return str(t).lower()

    return ".".join(parts)


def column_ref(c, alias_to_table: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """
    Return (alias, column_name) if c is a valid sqlglot Column.
    """
    if not HAS_SQLGLOT:
        return None

    if not isinstance(c, exp.Column):
        return None

    col = c.name
    if not col:
        return None

    alias = c.table
    if alias:
        alias = alias.lower()
    else:
        if len(alias_to_table) == 1:
            alias = next(iter(alias_to_table.keys()))
        else:
            return None

    if alias not in alias_to_table:
        return None

    return alias, col.lower()


def iter_comparisons(expr_node) -> Iterable:
    if not HAS_SQLGLOT or expr_node is None:
        return []

    results = []
    for cls in COMPARISON_CLASSES:
        results.extend(list(expr_node.find_all(cls)))
    return results


def comparison_operator(comp) -> str:
    if not HAS_SQLGLOT:
        return "other"

    if isinstance(comp, exp.EQ):
        return "="
    if isinstance(comp, exp.GT):
        return ">"
    if isinstance(comp, exp.GTE):
        return ">="
    if isinstance(comp, exp.LT):
        return "<"
    if isinstance(comp, exp.LTE):
        return "<="
    if isinstance(comp, exp.Like):
        return "like"
    if isinstance(comp, exp.In):
        return "in"
    if isinstance(comp, exp.Between):
        return "between"
    if isinstance(comp, exp.NEQ):
        return "!="
    return "other"


def comparison_sides(comp):
    """
    Return left/right expressions for common SQLGlot comparison nodes.
    """
    if isinstance(comp, exp.Between):
        return comp.this, None
    if isinstance(comp, exp.In):
        return comp.this, None

    left = getattr(comp, "left", None)
    right = getattr(comp, "right", None)

    if left is None:
        left = comp.args.get("this")
    if right is None:
        right = comp.args.get("expression")

    return left, right


def add_join_relation(
    G: nx.MultiDiGraph,
    alias_to_table: Dict[str, str],
    a1: str,
    c1: str,
    a2: str,
    c2: str,
):
    t1 = alias_to_table.get(a1, a1)
    t2 = alias_to_table.get(a2, a2)

    table1 = table_node_id(a1)
    table2 = table_node_id(a2)

    col1 = ensure_column_node(G, a1, c1, t1)
    col2 = ensure_column_node(G, a2, c2, t2)

    join_token = f"join::{a1}.{c1}={a2}.{c2}"

    # table-level join topology
    add_bidirectional_edge(
        G,
        table1,
        table2,
        edge_type="JOIN_TABLE",
        edge_token=join_token,
        left=f"{a1}.{c1}",
        right=f"{a2}.{c2}",
    )

    # column-level join topology
    add_bidirectional_edge(
        G,
        col1,
        col2,
        edge_type="JOIN_COLUMN",
        edge_token=join_token,
        left=f"{a1}.{c1}",
        right=f"{a2}.{c2}",
    )


def add_filter_relation(
    G: nx.MultiDiGraph,
    alias_to_table: Dict[str, str],
    alias: str,
    col: str,
    op: str,
    raw: str,
):
    table_name = alias_to_table.get(alias, alias)
    col_nid = ensure_column_node(G, alias, col, table_name)

    pred_nid = predicate_node_id(alias, col, op, raw)
    pred_token = f"predicate::{alias}.{col}:{op}"

    add_node(
        G,
        pred_nid,
        node_type="PREDICATE",
        token=pred_token,
        label=f"{alias}.{col} {op}",
        alias=alias,
        column=col,
        operator=op,
        raw=raw,
    )

    add_bidirectional_edge(
        G,
        col_nid,
        pred_nid,
        edge_type=comparison_to_edge_type(op),
        edge_token=pred_token,
    )


def sql_to_graph_sqlglot(sql: str) -> nx.MultiDiGraph:
    """
    SQLGlot-based SQL graph extraction.

    Nodes:
      - TABLE: table alias / relation
      - COLUMN: alias.column
      - PREDICATE: filter predicate on a column

    Edges:
      - TABLE_COLUMN: table -> column
      - JOIN_TABLE: table alias -> table alias
      - JOIN_COLUMN: join column -> join column
      - FILTER_*: column -> predicate
      - SELF: self loop
    """
    sql = normalize_sql(sql)
    G = nx.MultiDiGraph()

    try:
        tree = sqlglot.parse_one(sql)
    except Exception:
        return sql_to_graph_regex(sql)

    alias_to_table: Dict[str, str] = {}

    # 1. Tables
    for t in tree.find_all(exp.Table):
        full_name = table_full_name(t)
        alias = t.alias_or_name
        alias = alias.lower() if alias else full_name

        alias_to_table[alias] = full_name
        ensure_table_node(G, alias, full_name)

    # If SQL has no table parsed
    if not alias_to_table:
        add_node(
            G,
            "T::unknown",
            node_type="TABLE",
            token="table::unknown",
            label="unknown",
        )
        add_self_loops(G)
        return G

    # 2. JOIN ON conditions
    for j in tree.find_all(exp.Join):
        on_expr = j.args.get("on")
        for comp in iter_comparisons(on_expr):
            op = comparison_operator(comp)
            left, right = comparison_sides(comp)

            lref = column_ref(left, alias_to_table)
            rref = column_ref(right, alias_to_table)

            if lref and rref:
                a1, c1 = lref
                a2, c2 = rref
                if a1 != a2:
                    add_join_relation(G, alias_to_table, a1, c1, a2, c2)
                else:
                    raw = comp.sql(dialect="postgres") if hasattr(comp, "sql") else str(comp)
                    add_filter_relation(G, alias_to_table, a1, c1, op, raw)

    # 3. WHERE conditions: implicit joins and filters
    where_expr = tree.find(exp.Where)
    if where_expr is not None:
        for comp in iter_comparisons(where_expr):
            op = comparison_operator(comp)
            left, right = comparison_sides(comp)

            lref = column_ref(left, alias_to_table)
            rref = column_ref(right, alias_to_table)

            raw = comp.sql(dialect="postgres") if hasattr(comp, "sql") else str(comp)

            if lref and rref:
                a1, c1 = lref
                a2, c2 = rref
                if a1 != a2:
                    add_join_relation(G, alias_to_table, a1, c1, a2, c2)
                else:
                    add_filter_relation(G, alias_to_table, a1, c1, op, raw)
            elif lref:
                a, c = lref
                add_filter_relation(G, alias_to_table, a, c, op, raw)
            elif rref:
                a, c = rref
                add_filter_relation(G, alias_to_table, a, c, op, raw)

    # 4. Ensure isolated table nodes still have self-loops
    add_self_loops(G)

    return G


# ============================================================
# 5. Regex fallback parser
# ============================================================

def sql_to_graph_regex(sql: str) -> nx.MultiDiGraph:
    """
    Fallback parser when sqlglot is unavailable or fails.
    This is less accurate than sqlglot but still constructs real join edges.
    """
    sql_lower = normalize_sql(sql).lower()
    G = nx.MultiDiGraph()

    alias_to_table: Dict[str, str] = {}

    table_pattern = re.findall(
        r"(?:from|join)\s+([a-zA-Z0-9_.\"]+)(?:\s+(?:as\s+)?([a-zA-Z0-9_\"]+))?",
        sql_lower,
    )

    for tbl_name, alias in table_pattern:
        tbl_name = tbl_name.strip('"').lower()
        alias = alias.strip('"').lower() if alias else tbl_name
        if alias in {"on", "where", "join", "left", "right", "inner", "outer"}:
            alias = tbl_name

        alias_to_table[alias] = tbl_name
        ensure_table_node(G, alias, tbl_name)

    if not alias_to_table:
        add_node(
            G,
            "T::unknown",
            node_type="TABLE",
            token="table::unknown",
            label="unknown",
        )
        add_self_loops(G)
        return G

    # alias.col = alias.col as join
    join_pattern = re.findall(
        r"([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\s*=\s*([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)",
        sql_lower,
    )

    for a1, c1, a2, c2 in join_pattern:
        if a1 in alias_to_table and a2 in alias_to_table and a1 != a2:
            add_join_relation(G, alias_to_table, a1, c1, a2, c2)

    # filters: alias.col OP literal / placeholder
    filter_pattern = re.findall(
        r"([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\s*(=|>=|<=|>|<|like|in)\s*([^\s\)]+)",
        sql_lower,
    )

    for alias, col, op, rhs in filter_pattern:
        if alias in alias_to_table:
            # avoid treating column=column join as filter
            if "." in rhs:
                continue
            add_filter_relation(
                G,
                alias_to_table,
                alias,
                col,
                op,
                raw=f"{alias}.{col} {op} {rhs}",
            )

    add_self_loops(G)
    return G


def sql_to_graph(sql: str) -> nx.MultiDiGraph:
    if HAS_SQLGLOT:
        return sql_to_graph_sqlglot(sql)
    return sql_to_graph_regex(sql)


# ============================================================
# 6. Vocab building and PyG conversion
# ============================================================

def build_vocabs(graphs: List[nx.MultiDiGraph]) -> GraphVocabs:
    node_vocab = Vocab()
    edge_vocab = Vocab()

    for G in graphs:
        for _, attr in G.nodes(data=True):
            node_vocab.add(attr.get("token", "<UNK>"))

        for _, _, attr in G.edges(data=True):
            edge_vocab.add(attr.get("edge_token", "<UNK>"))

    return GraphVocabs(
        node_token_vocab=node_vocab,
        edge_token_vocab=edge_vocab,
    )


def nx_to_pyg(G: nx.MultiDiGraph, vocabs: GraphVocabs) -> Data:
    nodes = list(G.nodes)
    mapping = {n: i for i, n in enumerate(nodes)}

    node_type_ids = []
    node_token_ids = []

    for n in nodes:
        attr = G.nodes[n]
        node_type = attr.get("node_type", "UNKNOWN")
        token = attr.get("token", "<UNK>")

        node_type_ids.append(NODE_TYPES.get(node_type, NODE_TYPES["UNKNOWN"]))
        node_token_ids.append(vocabs.node_token_vocab.get(token))

    edge_index = []
    edge_type_ids = []
    edge_token_ids = []

    for u, v, attr in G.edges(data=True):
        if u not in mapping or v not in mapping:
            continue

        edge_index.append([mapping[u], mapping[v]])

        e_type = attr.get("edge_type", "SELF")
        e_token = attr.get("edge_token", "<UNK>")

        edge_type_ids.append(EDGE_TYPES.get(e_type, EDGE_TYPES["FILTER_OTHER"]))
        edge_token_ids.append(vocabs.edge_token_vocab.get(e_token))

    if len(edge_index) == 0:
        # This should rarely happen because self-loops are added.
        edge_index = [[0, 0]]
        edge_type_ids = [EDGE_TYPES["SELF"]]
        edge_token_ids = [vocabs.edge_token_vocab.get("SELF")]

    data = Data(
        node_type=torch.tensor(node_type_ids, dtype=torch.long),
        node_token=torch.tensor(node_token_ids, dtype=torch.long),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_type=torch.tensor(edge_type_ids, dtype=torch.long),
        edge_token=torch.tensor(edge_token_ids, dtype=torch.long),
        num_nodes=len(nodes),
    )

    return data


# ============================================================
# 7. Edge-aware SQL GNN Encoder
# ============================================================

class EdgeAwareSQLGNNEncoder(nn.Module):
    """
    Edge-aware SQL graph encoder.

    This encoder uses:
      - node type embedding
      - node token embedding
      - edge type embedding
      - edge token embedding

    GINEConv consumes edge attributes, unlike vanilla GCNConv.
    """

    def __init__(
        self,
        num_node_tokens: int,
        num_edge_tokens: int,
        hidden_dim: int = 64,
        out_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.node_type_emb = nn.Embedding(len(NODE_TYPES), hidden_dim)
        self.node_token_emb = nn.Embedding(num_node_tokens, hidden_dim)

        self.edge_type_emb = nn.Embedding(len(EDGE_TYPES), hidden_dim)
        self.edge_token_emb = nn.Embedding(num_edge_tokens, hidden_dim)

        mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.conv1 = GINEConv(mlp1, edge_dim=hidden_dim)
        self.conv2 = GINEConv(mlp2, edge_dim=hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def encode_nodes(self, data: Data) -> torch.Tensor:
        x = self.node_type_emb(data.node_type) + self.node_token_emb(data.node_token)

        edge_attr = (
            self.edge_type_emb(data.edge_type)
            + self.edge_token_emb(data.edge_token)
        )

        h1 = self.conv1(x, data.edge_index, edge_attr)
        h1 = self.norm1(torch.relu(h1) + x)

        h2 = self.conv2(h1, data.edge_index, edge_attr)
        h2 = self.norm2(torch.relu(h2) + h1)

        return h2

    def forward(self, data: Data) -> torch.Tensor:
        node_h = self.encode_nodes(data)
        graph_h = global_mean_pool(node_h, data.batch)
        return self.proj(graph_h)


# ============================================================
# 8. Diagnostics
# ============================================================

def graph_diagnostics(graphs: List[nx.MultiDiGraph], ids: List[str]) -> pd.DataFrame:
    rows = []
    for tid, G in zip(ids, graphs):
        num_nodes = G.number_of_nodes()
        num_edges_total = G.number_of_edges()

        edge_type_count = {k: 0 for k in EDGE_TYPES}
        node_type_count = {k: 0 for k in NODE_TYPES}

        for _, attr in G.nodes(data=True):
            nt = attr.get("node_type", "UNKNOWN")
            if nt in node_type_count:
                node_type_count[nt] += 1

        for _, _, attr in G.edges(data=True):
            et = attr.get("edge_type", "SELF")
            if et in edge_type_count:
                edge_type_count[et] += 1

        rows.append({
            "template_id": tid,
            "num_nodes": num_nodes,
            "num_edges_total": num_edges_total,
            "num_table_nodes": node_type_count["TABLE"],
            "num_column_nodes": node_type_count["COLUMN"],
            "num_predicate_nodes": node_type_count["PREDICATE"],
            "num_join_table_edges": edge_type_count["JOIN_TABLE"],
            "num_join_column_edges": edge_type_count["JOIN_COLUMN"],
            "num_filter_edges": (
                edge_type_count["FILTER_EQ"]
                + edge_type_count["FILTER_RANGE"]
                + edge_type_count["FILTER_LIKE"]
                + edge_type_count["FILTER_IN"]
                + edge_type_count["FILTER_OTHER"]
            ),
            "num_self_edges": edge_type_count["SELF"],
        })

    return pd.DataFrame(rows)


# ============================================================
# 9. Checkpoint loading
# ============================================================

def load_encoder_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    prefix: Optional[str] = None,
):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    if prefix:
        prefix = prefix.rstrip(".") + "."
        state = {
            k[len(prefix):]: v
            for k, v in state.items()
            if k.startswith(prefix)
        }

    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"[Checkpoint] Loaded from: {checkpoint_path}")
    if missing:
        print(f"[Checkpoint] Missing keys: {missing}")
    if unexpected:
        print(f"[Checkpoint] Unexpected keys: {unexpected}")


# ============================================================
# 10. Main export pipeline
# ============================================================

def export_embeddings(
    csv_path: str,
    out_dir: str,
    id_col: str = "template_id",
    sql_col: str = "template_sql",
    hidden_dim: int = 64,
    out_dim: int = 32,
    batch_size: int = 32,
    checkpoint: Optional[str] = None,
    checkpoint_prefix: Optional[str] = None,
    allow_random: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if id_col not in df.columns:
        raise ValueError(f"Cannot find id column: {id_col}")
    if sql_col not in df.columns:
        raise ValueError(f"Cannot find SQL column: {sql_col}")

    ids = df[id_col].astype(str).tolist()
    sqls = df[sql_col].astype(str).tolist()

    print(f"[Info] SQLGlot available: {HAS_SQLGLOT}")
    print(f"[Info] Loading templates: {len(sqls)}")

    graphs = []
    for sql in sqls:
        G = sql_to_graph(sql)
        graphs.append(G)

    # Diagnostics
    diag = graph_diagnostics(graphs, ids)
    diag_path = os.path.join(out_dir, "template_graph_diagnostics.csv")
    diag.to_csv(diag_path, index=False)
    print(f"[Info] Saved graph diagnostics to: {diag_path}")

    print("[Diagnostics Summary]")
    print(diag[[
        "num_nodes",
        "num_edges_total",
        "num_join_table_edges",
        "num_join_column_edges",
        "num_filter_edges",
    ]].describe())

    # Warn if too many graphs have no join edges
    no_join_ratio = (diag["num_join_table_edges"] == 0).mean()
    print(f"[Info] Ratio of templates without join-table edges: {no_join_ratio:.2%}")

    # Build vocabs
    vocabs = build_vocabs(graphs)
    vocab_path = os.path.join(out_dir, "template_graph_vocabs.json")
    vocabs.save(vocab_path)
    print(f"[Info] Saved vocabs to: {vocab_path}")
    print(f"[Info] Node token vocab size: {len(vocabs.node_token_vocab)}")
    print(f"[Info] Edge token vocab size: {len(vocabs.edge_token_vocab)}")

    data_list = []
    for G in graphs:
        data = nx_to_pyg(G, vocabs)
        data_list.append(data)

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EdgeAwareSQLGNNEncoder(
        num_node_tokens=len(vocabs.node_token_vocab),
        num_edge_tokens=len(vocabs.edge_token_vocab),
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        dropout=0.0,
    ).to(device)

    if checkpoint:
        load_encoder_checkpoint(
            model,
            checkpoint_path=checkpoint,
            device=device,
            prefix=checkpoint_prefix,
        )
    else:
        if not allow_random:
            raise RuntimeError(
                "\n[Error] No trained checkpoint is provided.\n"
                "Do not export random GNN embeddings for paper experiments.\n"
                "Either:\n"
                "  1) train this SQL graph encoder end-to-end inside the forecasting model, then pass --checkpoint; or\n"
                "  2) explicitly pass --allow-random only for debugging.\n"
            )
        print(
            "[Warning] Exporting embeddings from a randomly initialized GNN. "
            "Use this only for debugging, not for paper experiments."
        )

    model.eval()
    embeddings = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = model(batch)
            embeddings.append(emb.cpu().numpy())

    embeddings = np.vstack(embeddings)

    npy_path = os.path.join(out_dir, "template_gnn_embeddings.npy")
    np.save(npy_path, embeddings)

    emb_cols = [f"gnn_{i}" for i in range(embeddings.shape[1])]
    out_df = pd.DataFrame({id_col: ids})
    out_df = pd.concat([out_df, pd.DataFrame(embeddings, columns=emb_cols)], axis=1)

    csv_out_path = os.path.join(out_dir, "template_gnn_embeddings.csv")
    out_df.to_csv(csv_out_path, index=False)

    print(f"[Info] Saved embeddings shape: {embeddings.shape}")
    print(f"[Info] Saved npy to: {npy_path}")
    print(f"[Info] Saved csv to: {csv_out_path}")


# ============================================================
# 11. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build edge-aware SQL structural graphs and export GNN embeddings."
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="../processed/SDSS/0.1sampling/template_param_string_dict_modified.csv",
        help="Input CSV file containing template_id and template_sql.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="../processed/SDSS/0.1sampling/",
        help="Output directory.",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="template_id",
        help="Template ID column name.",
    )
    parser.add_argument(
        "--sql-col",
        type=str,
        default="template_sql",
        help="SQL template column name.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension of SQL GNN encoder.",
    )
    parser.add_argument(
        "--out-dim",
        type=int,
        default=32,
        help="Output dimension of SQL GNN embedding.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained SQL GNN checkpoint or forecasting model checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default=None,
        help=(
            "Prefix of SQL GNN parameters inside a full forecasting checkpoint, "
            "e.g., 'sql_gnn' or 'model.sql_gnn'."
        ),
    )

    parser.add_argument(
        "--allow-random",
        action="store_true",
        default=True,
        help="Allow exporting randomly initialized embeddings. Debug only.",
    )

    parser.add_argument(
        "--no-random",
        action="store_false",
        dest="allow_random",
        help="Disable random initialized embedding export and require a checkpoint.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    export_embeddings(
        csv_path=args.csv,
        out_dir=args.out_dir,
        id_col=args.id_col,
        sql_col=args.sql_col,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        batch_size=args.batch_size,
        checkpoint=args.checkpoint,
        checkpoint_prefix=args.checkpoint_prefix,
        allow_random=args.allow_random,
    )


if __name__ == "__main__":
    main()