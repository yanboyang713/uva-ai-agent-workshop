from __future__ import annotations

import random
import re
import sqlite3
from collections import Counter


def extract_sql(raw: str) -> str:
    """
    Extract SQL from model output. The base model often appends Answer:, Explanation:,
    or hallucinated query results. Take only the SQL part (before those markers).
    """
    if not raw:
        return ""
    for sep in ["\nAnswer:", "\nExplanation:", "\nSQL:"]:
        idx = raw.find(sep)
        if idx >= 0:
            raw = raw[:idx]
    return raw.strip()


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL for comparison:
    - Strip <|end_of_text|> and similar special tokens
    - Treat single and double quotes as equivalent
    - Normalize whitespace
    """
    if not sql:
        return ""
    sql = re.sub(r"<\|[^|]+\|>", "", sql)
    sql = sql.strip()
    sql = sql.replace('"', "'")
    sql = " ".join(sql.split())
    sql = sql.rstrip(";")
    return sql


def _extract_literals(sql: str) -> tuple[list[str], list[float]]:
    """Extract string and numeric literals from a SQL query."""
    strings = re.findall(r"'([^']*)'", sql) + re.findall(r'"([^"]*)"', sql)
    numbers = [
        float(m.group(1))
        for m in re.finditer(r"(?<![a-zA-Z_])(\d+(?:\.\d+)?)(?![a-zA-Z_])", sql)
        if m.group(1)
    ]
    return strings, numbers


def _build_db(
    schema: str,
    str_lits: list[str],
    num_lits: list[float],
    seed: int = 0,
    n_rows: int = 50,
) -> sqlite3.Connection:
    """Build an in-memory SQLite DB from schema, seeded with query literals."""
    rng = random.Random(seed)
    nocase = re.sub(
        r"\b(\w+(?:\(\d+\))?)\s*(?=,|\))",
        lambda m: m.group(0) + " COLLATE NOCASE"
        if any(k in m.group(1).upper() for k in ("VARCHAR", "TEXT", "CHAR"))
        else m.group(0),
        schema,
    )

    conn = sqlite3.connect(":memory:")
    for stmt in [s.strip() for s in nocase.split(";") if s.strip()]:
        if not stmt.upper().startswith("CREATE"):
            continue
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            continue

        header = re.search(r"CREATE\s+TABLE\s+(\w+)\s*\((.+)\)", stmt, re.IGNORECASE | re.DOTALL)
        if not header:
            continue

        table_name = header.group(1)
        pools: list[list[object]] = []
        for part in header.group(2).split(","):
            tokens = part.strip().split()
            if len(tokens) < 2:
                continue
            if tokens[0].upper().startswith(("PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT")):
                continue

            is_numeric = any(
                key in tokens[1].upper() for key in ("INT", "REAL", "FLOAT", "DOUBLE", "NUMERIC")
            )
            if is_numeric:
                values: list[object] = list(set(num_lits)) + [float(i * 7 + len(pools) * 3 + 1) for i in range(8)]
            else:
                values = list(set(str_lits)) + [f"{tokens[0]}_v{i}" for i in range(6)]
            rng.shuffle(values)
            pools.append(values)

        for row_idx in range(n_rows):
            values = [pool[row_idx % len(pool)] for pool in pools]
            try:
                conn.execute(f"INSERT INTO {table_name} VALUES ({','.join('?' * len(values))})", values)
            except sqlite3.Error:
                break

    conn.commit()
    return conn


def _exec_match(
    generated_sql: str,
    expected_sql: str,
    schema: str,
    strings: list[str],
    numbers: list[float],
    seed: int,
) -> bool:
    conn = _build_db(schema, strings, numbers, seed=seed)
    try:
        generated_rows = conn.execute(generated_sql).fetchall()
        expected_rows = conn.execute(expected_sql).fetchall()
    except sqlite3.Error:
        return False
    finally:
        conn.close()

    if not generated_rows and not expected_rows:
        return True
    if generated_rows and expected_rows and len(generated_rows[0]) != len(expected_rows[0]):
        return False
    return Counter(generated_rows) == Counter(expected_rows)


def sql_matches(generated: str, expected: str | list[str], schema: str = "") -> bool:
    """
    Check if generated SQL matches expected.
    With schema: execution-based (runs both on multiple seeded SQLite DBs).
    Without schema: normalized string comparison.
    """
    generated_sql = normalize_sql(extract_sql(generated))
    if not generated_sql:
        return False

    expected_list = [expected] if isinstance(expected, str) else expected
    if not schema:
        return any(generated_sql == normalize_sql(item) for item in expected_list)

    strings: list[str] = []
    numbers: list[float] = []
    for sql in [generated_sql] + [normalize_sql(item) for item in expected_list]:
        str_lits, num_lits = _extract_literals(sql)
        strings.extend(str_lits)
        numbers.extend(num_lits)

    for candidate in expected_list:
        normalized = normalize_sql(candidate)
        if all(
            _exec_match(generated_sql, normalized, schema, strings, numbers, seed=idx * 97 + 13)
            for idx in range(5)
        ):
            return True
    return False

