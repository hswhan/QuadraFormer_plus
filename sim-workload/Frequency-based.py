import os
import re
import argparse
from collections import Counter, defaultdict


SQL_KEYWORDS = {
    "and", "or", "where", "on", "select", "from", "join", "left", "right",
    "inner", "outer", "full", "cross", "group", "order", "by", "having",
    "limit", "offset", "union", "as", "in", "like", "between", "is", "null",
    "not", "exists", "case", "when", "then", "else", "end"
}


def normalize_sql(sql: str) -> str:
    sql = re.sub(r"--.*?$", " ", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = sql.replace("\n", " ")
    sql = re.sub(r"\s+", " ", sql)
    return sql.strip().lower()


def strip_identifier(x: str) -> str:
    return x.strip().strip('"').strip("`").strip("[").strip("]")


def extract_from_zone(sql_lower: str) -> str:
    """
    提取 FROM 后、WHERE/GROUP/ORDER/LIMIT 等之前的区域。
    """
    m = re.search(r"\bfrom\b", sql_lower)
    if not m:
        return ""

    start = m.end()
    end_match = re.search(
        r"\bwhere\b|\bgroup\s+by\b|\border\s+by\b|\bhaving\b|\blimit\b|\boffset\b|\bunion\b",
        sql_lower[start:],
    )

    if end_match:
        end = start + end_match.start()
    else:
        end = len(sql_lower)

    return sql_lower[start:end].strip()


def extract_alias_table_map(sql_lower: str):
    """
    从 FROM 区域提取 alias -> table。
    支持：
      FROM table a, table2 b
      FROM table AS a JOIN table2 AS b ON ...
      FROM schema.table a
    """
    from_zone = extract_from_zone(sql_lower)
    alias_to_table = {}

    if not from_zone:
        return alias_to_table

    # 匹配 FROM 区域里的主表、逗号表、JOIN 表
    # 例：
    #   table_a a
    #   schema.table_b AS b
    #   JOIN table_c c
    table_pattern = re.compile(
        r"(?:^|,|\bjoin\s+)\s*([a-z_][a-z0-9_\.]*)"
        r"(?:\s+(?:as\s+)?([a-z_][a-z0-9_]*))?",
        flags=re.IGNORECASE,
    )

    for m in table_pattern.finditer(from_zone):
        table = strip_identifier(m.group(1))
        alias = m.group(2)

        # 排除子查询、关键字、空表
        if not table or table in SQL_KEYWORDS:
            continue

        if alias:
            alias = strip_identifier(alias)
            if alias in SQL_KEYWORDS:
                alias = table.split(".")[-1]
        else:
            alias = table.split(".")[-1]

        alias_to_table[alias] = table

    return alias_to_table


def extract_condition_zone(sql_lower: str) -> str:
    """
    提取 WHERE 和 ON 条件区域，避免统计 SELECT 里的列。
    """
    parts = []

    # WHERE 条件
    m = re.search(r"\bwhere\b", sql_lower)
    if m:
        start = m.end()
        end_match = re.search(
            r"\bgroup\s+by\b|\border\s+by\b|\bhaving\b|\blimit\b|\boffset\b|\bunion\b",
            sql_lower[start:],
        )
        end = start + end_match.start() if end_match else len(sql_lower)
        parts.append(sql_lower[start:end])

    # ON 条件
    for on_m in re.finditer(r"\bon\b", sql_lower):
        start = on_m.end()
        end_match = re.search(
            r"\bjoin\b|\bwhere\b|\bgroup\s+by\b|\border\s+by\b|\bhaving\b|\blimit\b|\boffset\b|\bunion\b",
            sql_lower[start:],
        )
        end = start + end_match.start() if end_match else len(sql_lower)
        parts.append(sql_lower[start:end])

    # 如果没有 WHERE/ON，就不强行从 SELECT 里抓列
    return " ".join(parts)


def extract_index_columns_with_table(sql: str):
    """
    返回：
      resolved: set[(table, column)]
      unresolved: set[column]
    """
    sql_lower = normalize_sql(sql)

    alias_to_table = extract_alias_table_map(sql_lower)
    condition_zone = extract_condition_zone(sql_lower)

    resolved = set()
    unresolved = set()

    if not condition_zone:
        return resolved, unresolved

    # 1. alias.column
    alias_dot_cols = re.findall(r"\b([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\b", condition_zone)
    for alias, col in alias_dot_cols:
        if alias.isdigit() and col.isdigit():
            continue

        alias = strip_identifier(alias)
        col = strip_identifier(col)

        if alias in alias_to_table:
            resolved.add((alias_to_table[alias], col))
        else:
            # alias 没解析出来时，保留 alias 作为“疑似表名”
            resolved.add((alias, col))

    # 2. 无 alias 的纯列名
    # 注意 (?<!\.) 防止 a.col 里的 col 被重复匹配
    pure_cols = re.findall(
        r"(?<!\.)\b([a-z_][a-z0-9_]*)\s*(=|<=|>=|<>|!=|<|>|like|in|between|is)\b",
        condition_zone,
    )

    for col, _op in pure_cols:
        col = strip_identifier(col)

        if col in SQL_KEYWORDS:
            continue

        # 如果 FROM 里只有一张表，可以直接归属到该表
        unique_tables = sorted(set(alias_to_table.values()))
        if len(unique_tables) == 1:
            resolved.add((unique_tables[0], col))
        else:
            unresolved.add(col)

    return resolved, unresolved


def safe_idx_name(table: str, col: str) -> str:
    s = f"idx_{table}_{col}"
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s)
    return s[:60]


def main():
    parser = argparse.ArgumentParser(description="Frequency-based index baseline with table.column extraction")
    parser.add_argument("--sql-file", type=str, default="tiramisuworkload_baseline.sql")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if not os.path.exists(args.sql_file):
        print(f"[Error] 找不到 SQL 文件: {args.sql_file}")
        return

    print(f"[*] 分析工作负载: {args.sql_file}")

    table_col_counter = Counter()
    unresolved_counter = Counter()

    total_sqls = 0
    parsed_sqls = 0
    current_sql = ""

    with open(args.sql_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("--"):
                continue

            current_sql += " " + line

            if current_sql.endswith(";"):
                total_sqls += 1

                resolved, unresolved = extract_index_columns_with_table(current_sql)

                if resolved or unresolved:
                    parsed_sqls += 1

                for table, col in resolved:
                    table_col_counter[(table, col)] += 1

                for col in unresolved:
                    unresolved_counter[col] += 1

                current_sql = ""

    print(f"[*] 分析完成! 总 SQL 数: {total_sqls}, 成功提取过滤/连接列的 SQL 数: {parsed_sqls}")
    print("-" * 70)

    if not table_col_counter and not unresolved_counter:
        print("[!] 没有提取到任何候选索引列。")
        return

    print(f"🏆 Top-{args.top_k} table.column 频率基线推荐:")
    for rank, ((table, col), count) in enumerate(table_col_counter.most_common(args.top_k), 1):
        idx_name = safe_idx_name(table, col)
        print(f"  {rank}. {table}.{col:<30}  出现 {count} 次")
        print(f"      CREATE INDEX {idx_name} ON {table} ({col});")

    if unresolved_counter:
        print("-" * 70)
        print("⚠️ 无法自动定位物理表的无前缀列:")
        for rank, (col, count) in enumerate(unresolved_counter.most_common(args.top_k), 1):
            print(f"  {rank}. {col:<30}  出现 {count} 次")
        print("这些列在 SQL 中没有 alias/table 前缀，且 FROM 中存在多张表；需要结合数据库 schema 或人工映射。")

    print("-" * 70)


if __name__ == "__main__":
    main()