#!/usr/bin/env python3
# make_baseline_sql_from_request_log_fixed.py
import os, json, re, argparse, ast
import pandas as pd
import datetime
import random
from typing import Dict, List, Set, Tuple

SIM_DIR = "."
REQ_CSV = os.path.join(SIM_DIR, "request_log.csv")
INSERT_SQL = os.path.join(SIM_DIR, "insert.sql")
TEMPLATE_CANDIDATES = [
    os.path.join(SIM_DIR, "template_param_string_dict_modified.csv"),
    os.path.join("tiramisu", "template_param_string_dict_modified.csv"),
]
OUT_SQL = "tiramisuworkload_baseline.sql"
OUT_SELECT_SQL = "tiramisuworkload_baseline_selects.sql"
SKIPPED_FILE = "skipped_examples.txt"
INSERT_DATA_CACHE = "insert_data_cache.json"

# ====== Placeholder filling strategy: 'one' → replace all with 1; 'rand5' → random 1..5 per occurrence ======
C_FILL_MODE = "one"  # Change to "rand5" for random values between 1 and 5

SQL_KEYWORDS = set(x.upper() for x in """
SELECT FROM WHERE JOIN ON GROUP ORDER HAVING LIMIT DISTINCT AS AND OR NOT IN IS NULL EXISTS BETWEEN
""".split())


def _c_fill_value_str() -> str:
    if C_FILL_MODE.lower() == "rand5":
        return str(random.randint(1, 5))
    return "1"

def wipe_placeholders_to_num(s: str) -> str:
    """
    Replace any remaining placeholders with numeric values:
    - c\d+  → number
    - $number → number (PostgreSQL positional parameters)
    - :name → number (skip :: type casts; only match : followed by letter/underscore)
    """
    if not s:
        return s

    # c1/c2/... case-insensitive
    s = re.sub(r'(?i)\bc\d+\b', lambda m: _c_fill_value_str(), s)

    # $1/$2 positional parameters
    s = re.sub(r'\$\d+\b', lambda m: _c_fill_value_str(), s)

    # :name named parameters (avoid matching ::)
    s = re.sub(r'(?<!:):[A-Za-z_]\w*\b', lambda m: _c_fill_value_str(), s)

    return s


def parse_insert_sql(insert_sql_path: str) -> Dict[str, Dict[str, Set[str]]]:
    if os.path.exists(INSERT_DATA_CACHE):
        with open(INSERT_DATA_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)

    insert_data: Dict[str, Dict[str, Set[str]]] = {}
    if not os.path.exists(insert_sql_path):
        raise FileNotFoundError(f"insert.sql not found at {insert_sql_path}")

    with open(insert_sql_path, "r", encoding="utf-8") as f:
        sql_content = f.read()

    insert_pattern = re.compile(
        r"INSERT INTO\s+`?([\w\.]+)`?\s*\((.*?)\)\s*VALUES\s*(.+?);",
        re.DOTALL | re.IGNORECASE
    )
    for match in insert_pattern.finditer(sql_content):
        table_name = match.group(1).lower().strip()
        cols_str = match.group(2).strip()
        values_str = match.group(3).strip()

        columns = [col.strip(" `\"'") for col in re.split(r',\s*', cols_str)]
        row_pattern = re.compile(r"\((.*?)\)(?:,|$)", re.DOTALL)
        for row_match in row_pattern.finditer(values_str):
            row_text = row_match.group(1).strip()
            values = []
            in_quote = False
            current_val = []
            quote_char = None
            for c in row_text:
                if c in ["'", '"'] and (not quote_char or c == quote_char):
                    in_quote = not in_quote
                    quote_char = c if in_quote else None
                    current_val.append(c)
                elif c == ',' and not in_quote:
                    values.append(''.join(current_val).strip())
                    current_val = []
                else:
                    current_val.append(c)
            if current_val:
                values.append(''.join(current_val).strip())

            if len(values) != len(columns):
                continue
            for col, val in zip(columns, values):
                val_clean = val.strip("'\"")
                if table_name not in insert_data:
                    insert_data[table_name] = {}
                if col not in insert_data[table_name]:
                    insert_data[table_name][col] = set()
                insert_data[table_name][col].add(val_clean)

    with open(INSERT_DATA_CACHE, "w", encoding="utf-8") as f:
        json.dump(
            {k: {k2: list(v2) for k2, v2 in v.items()} for k, v in insert_data.items()},
            f,
            ensure_ascii=False
        )
    return insert_data


def get_table_from_sql(sql: str) -> str:
    sql_clean = re.sub(r'\s+', ' ', sql.strip())
    from_match = re.search(r'\bFROM\s+`?([\w\.]+)`?', sql_clean, re.IGNORECASE)
    if from_match:
        return from_match.group(1).lower().strip()
    join_match = re.search(r'\bJOIN\s+`?([\w\.]+)`?', sql_clean, re.IGNORECASE)
    if join_match:
        return join_match.group(1).lower().strip()
    return ""


def find_template_csv():
    for p in TEMPLATE_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def load_templates(path: str) -> Dict[str, Dict]:
    df = pd.read_csv(path, dtype=str).fillna("")
    if "template_id" not in df.columns or "template_sql" not in df.columns:
        raise ValueError("Template CSV must contain template_id and template_sql columns.")
    templates = {}
    for _, r in df.iterrows():
        tid = str(r["template_id"])
        templates[tid] = {
            "sql": str(r["template_sql"]),
            "param_types": r.get("param_types", "") if "param_types" in r.index else "",
            "string_dict": r.get("string_dict", "") if "string_dict" in r.index else "",
            "table": get_table_from_sql(str(r["template_sql"]))
        }
    return templates


def format_value_for_sql(val):
    if isinstance(val, bool):
        return "true" if val else "false"
    if val is None:
        return "NULL"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        s = f"{val:.6f}".rstrip('0').rstrip('.')
        return s
    if isinstance(val, str):
        s = val.replace("'", "''").replace('"', '""')
        return f"'{s}'"
    s = str(val).replace("'", "''").replace('"', '""')
    return f"'{s}'"


def find_bare_placeholders(sql: str) -> List[str]:
    tokens = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', sql)
    cand = []
    for t in tokens:
        tu = t.upper()
        if tu in SQL_KEYWORDS:
            continue
        if re.match(r'^[cp][0-9]+$', t) or re.match(r'^[a-z][0-9]+$', t) or re.match(r'^[a-z]+[0-9]+$', t):
            cand.append(t)
    seen = set()
    out = []
    for c in cand:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def get_real_value(table: str, col: str, insert_data: Dict[str, Dict[str, Set[str]]], current_val) -> str:
    if not table or table not in insert_data:
        return current_val
    table_data = insert_data[table]
    if col not in table_data:
        return current_val
    valid_values = table_data[col]
    if not valid_values:
        return current_val
    return random.choice(list(valid_values))


def substitute_params_into_sql(
        template_sql: str,
        params: Dict,
        tpl_meta: Dict,
        insert_data: Dict[str, Dict[str, Set[str]]]
) -> str:
    s = template_sql
    table = tpl_meta.get("table", "")

    # 1) Replace {name}
    for k, v in (params or {}).items():
        try:
            s = s.replace("{" + k + "}", format_value_for_sql(v))
        except Exception:
            pass

    # 2) Build mappings from "column = placeholder / placeholder = column"
    def _extract_col_ph_pairs(sql: str) -> List[Tuple[str, str]]:
        pairs = []
        for m in re.finditer(r'(?i)([`"]?[a-z_][\w`"]*(?:\.[`"]?[a-z_][\w`"]*)?)\s*=\s*(\{[\w]+\}|c\d+|:\w+|\$\d+)', sql):
            left = m.group(1).replace('`', '').replace('"', '')
            col = left.split('.')[-1]
            ph = m.group(2)
            if ph.startswith('{') and ph.endswith('}'):
                ph = ph[1:-1]
            pairs.append((col, ph))
        for m in re.finditer(r'(?i)(\{[\w]+\}|c\d+|:\w+|\$\d+)\s*=\s*([`"]?[a-z_][\w`"]*(?:\.[`"]?[a-z_][\w`"]*)?)', sql):
            right = m.group(2).replace('`', '').replace('"', '')
            col = right.split('.')[-1]
            ph = m.group(1)
            if ph.startswith('{') and ph.endswith('}'):
                ph = ph[1:-1]
            pairs.append((col, ph))
        seen = set()
        out = []
        for col, ph in pairs:
            key = (col.lower(), ph)
            if key not in seen:
                seen.add(key)
                out.append((col, ph))
        return out

    col_ph_pairs = _extract_col_ph_pairs(s)

    ptypes = {}
    if tpl_meta.get("param_types"):
        try:
            ptypes = json.loads(tpl_meta["param_types"]) or {}
        except Exception:
            ptypes = {}

    # 3) Replace each pair (prefer params, then column values, fallback to real values from insert.sql)
    for col, ph in col_ph_pairs:
        if ph in params:
            val = params[ph]
        elif col in params:
            val = params[col]
        else:
            val = get_real_value(table, col, insert_data, None)
        if val is None:
            continue
        rep = format_value_for_sql(val)

        s = s.replace("{" + ph + "}", rep)
        s = re.sub(r'\b' + re.escape(ph) + r'\b', rep, s)

    # 4) Fallback: try to map bare placeholders using params order/single-value assumption
    bare = find_bare_placeholders(s)
    if bare:
        mapped = {}
        if ptypes:
            for b in bare:
                if b in params:
                    mapped[b] = params[b]
                    continue
                for pk, colname in ptypes.items():
                    if pk == b and isinstance(colname, str) and colname in params:
                        mapped[b] = params[colname]
                        break
        if not mapped:
            pkeys = list(params.keys())
            if len(bare) == len(pkeys) and bare:
                mapped = {b: params[pkeys[i]] for i, b in enumerate(bare)}
        if not mapped and len(params) == 1:
            single_val = next(iter(params.values()))
            mapped = {b: single_val for b in bare}
        for b, val in mapped.items():
            s = re.sub(r'\b' + re.escape(b) + r'\b', format_value_for_sql(val), s)

    # 5) Final fallback: replace any leftover cN / $N / :name with numbers
    s = wipe_placeholders_to_num(s)

    # 6) Syntax safety check
    if "{" in s or "}" in s:
        return ""
    if not re.match(r'^\s*select\b', s, flags=re.IGNORECASE):
        return ""
    return s


def make_baseline(last_hours: int = 48, dedupe: bool = False):
    insert_data = parse_insert_sql(INSERT_SQL)
    print(f"Loaded data from insert.sql covering {len(insert_data)} tables")

    if not os.path.exists(REQ_CSV):
        raise FileNotFoundError(f"{REQ_CSV} not found.")
    tpl_csv = find_template_csv()
    if not tpl_csv:
        raise FileNotFoundError("Template CSV not found.")
    templates = load_templates(tpl_csv)

    req = pd.read_csv(REQ_CSV, dtype=str).fillna("")
    if "ts" not in req.columns:
        raise ValueError("request_log.csv missing ts column")
    req["ts_dt"] = pd.to_datetime(req["ts"], errors="coerce")
    max_dt = req["ts_dt"].max()
    if pd.isna(max_dt):
        max_dt = datetime.datetime.fromtimestamp(os.path.getmtime(REQ_CSV))
    cutoff = max_dt - pd.Timedelta(hours=last_hours)
    sel = req[req["ts_dt"] >= cutoff].copy()
    sel["freq_num"] = pd.to_numeric(sel.get("freq", 1), errors="coerce").fillna(1).astype(int)

    concrete_sqls = []
    skipped = []
    for _, row in sel.iterrows():
        tid = str(row.get("template_id", ""))
        if tid not in templates:
            skipped.append((tid, "no_template", row.to_dict()))
            continue
        tpl_meta = templates[tid]
        tpl_sql = tpl_meta["sql"]
        params_raw = row.get("param_json", "")
        try:
            params = json.loads(params_raw) if params_raw else {}
        except Exception:
            try:
                params = eval(params_raw, {"null": None})
            except Exception:
                params = {}

        concrete = substitute_params_into_sql(tpl_sql, params, tpl_meta, insert_data)
        if not concrete:
            skipped.append((tid, tpl_sql, params, row.get("ts")))
            continue
        if not concrete.strip().endswith(";"):
            concrete = concrete.strip() + ";"

        # Extra cleaning pass (double safety)
        concrete = wipe_placeholders_to_num(concrete)
        concrete_sqls.append(concrete)

    if dedupe:
        seen = set()
        new = []
        for s in concrete_sqls:
            if s not in seen:
                seen.add(s)
                new.append(s)
        concrete_sqls = new

    # Final global cleaning pass (extreme safety)
    concrete_sqls = [wipe_placeholders_to_num(s) for s in concrete_sqls]

    with open(OUT_SQL, "w", encoding="utf-8") as f:
        for s in concrete_sqls:
            f.write(s + "\n")
    selects = [s for s in concrete_sqls if s.strip().lower().startswith("select")]
    with open(OUT_SELECT_SQL, "w", encoding="utf-8") as f:
        for s in selects:
            f.write(s + "\n")
    with open(SKIPPED_FILE, "w", encoding="utf-8") as f:
        for item in skipped[:200]:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

    print(
        f"Wrote {len(concrete_sqls)} statements to {OUT_SQL}; SELECT rows {len(selects)}; skipped {len(skipped)} (examples in {SKIPPED_FILE}).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=int, default=48)
    args = p.parse_args()
    if os.path.exists(INSERT_DATA_CACHE):
        os.remove(INSERT_DATA_CACHE)
    make_baseline(last_hours=args.hours)


if __name__ == "__main__":
    main()