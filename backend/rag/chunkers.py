"""STEP 1 - CHUNKING

Split source data into small pieces ("chunks") that can be embedded and retrieved.

Document strategies : fixed_size, sentence, paragraph
SQL strategies      : row, entity, schema
"""
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path

from backend import config


@dataclass
class Chunk:
    id: str
    text: str
    source: str            # file name or table name
    strategy: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Loading raw data
# --------------------------------------------------------------------------
def load_docs() -> dict[str, str]:
    """Return {file_name: text} for every markdown doc."""
    return {p.name: p.read_text() for p in sorted(Path(config.DOCS_DIR).glob("*.md"))}


def load_tables() -> dict[str, list[dict]]:
    """Return {table_name: [row_dict, ...]} for every SQLite table."""
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    data = {t: [dict(r) for r in con.execute(f"SELECT * FROM {t}")] for t in tables}
    con.close()
    return data


# --------------------------------------------------------------------------
# Document chunking
# --------------------------------------------------------------------------
def chunk_fixed_size(name: str, text: str, size: int = config.FIXED_CHUNK_SIZE,
                     overlap: int = config.FIXED_CHUNK_OVERLAP) -> list[Chunk]:
    """Cut every `size` characters, stepping back `overlap` characters each time.
    Simple, but it can cut a sentence (or a word) in half."""
    text = re.sub(r"\s+", " ", text).strip()
    step = max(1, size - overlap)
    chunks = []
    for i, start in enumerate(range(0, len(text), step)):
        piece = text[start:start + size]
        chunks.append(Chunk(f"{name}#fixed-{i}", piece, name, "fixed_size",
                            {"start": start, "end": start + len(piece)}))
        if start + size >= len(text):
            break
    return chunks


def chunk_sentence(name: str, text: str, n: int = config.SENTENCES_PER_CHUNK) -> list[Chunk]:
    """Group every `n` sentences. Never cuts a sentence in half."""
    body = re.sub(r"^#+ .*$", "", text, flags=re.M)          # drop headings
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", body)) if s.strip()]
    return [Chunk(f"{name}#sent-{i // n}", " ".join(sentences[i:i + n]), name, "sentence",
                  {"sentences": len(sentences[i:i + n])})
            for i in range(0, len(sentences), n)]


def chunk_paragraph(name: str, text: str) -> list[Chunk]:
    """Structure-aware: one chunk per '## heading' section, keeping the title and
    heading as context. Usually the best choice for well-structured docs."""
    title = (re.search(r"^# (.+)$", text, re.M) or [None, name])[1]
    sections = re.split(r"^## ", text, flags=re.M)[1:]
    chunks = []
    for i, sec in enumerate(sections):
        heading, _, body = sec.partition("\n")
        chunk_text = f"{title} - {heading.strip()}: {' '.join(body.split())}"
        chunks.append(Chunk(f"{name}#para-{i}", chunk_text, name, "paragraph",
                            {"heading": heading.strip()}))
    return chunks


DOC_STRATEGIES = {
    "fixed_size": chunk_fixed_size,
    "sentence": chunk_sentence,
    "paragraph": chunk_paragraph,
}


def chunk_docs(strategy: str, **params) -> list[Chunk]:
    fn = DOC_STRATEGIES[strategy]
    return [c for name, text in load_docs().items() for c in fn(name, text, **params)]


# --------------------------------------------------------------------------
# SQL chunking - turn structured rows into text an embedder can understand
# --------------------------------------------------------------------------
def _row_to_text(table: str, row: dict) -> str:
    if table == "products":
        stock = "in stock" if row["stock"] > 0 else "out of stock"
        return (f"Product {row['name']} ({row['category']}) costs {row['price']:.0f} dollars, "
                f"{stock} ({row['stock']} units). {row['description']}")
    if table == "customers":
        return f"Customer {row['name']} lives in {row['city']} and is a {row['tier']} tier member."
    if table == "orders":
        return (f"Order {row['id']}: customer {row['customer_id']} bought {row['quantity']} x "
                f"product {row['product_id']} on {row['order_date']}, status {row['status']}.")
    return ", ".join(f"{k}: {v}" for k, v in row.items())


def chunk_sql_row(con: sqlite3.Connection) -> list[Chunk]:
    """One row = one chunk. Good for lookups ('price of the smart watch')."""
    chunks = []
    for table in ("products", "customers", "orders"):
        for row in con.execute(f"SELECT * FROM {table}"):
            row = dict(row)
            chunks.append(Chunk(f"{table}#{row['id']}", _row_to_text(table, row), table, "row",
                                {"row": row}))
    return chunks


def chunk_sql_entity(con: sqlite3.Connection) -> list[Chunk]:
    """Entity-grouped: JOIN related tables so one chunk = one customer with all
    their orders. Answers 'what did Asha buy?' in a single chunk."""
    chunks = []
    for c in con.execute("SELECT * FROM customers"):
        orders = con.execute(
            """SELECT o.id, o.quantity, o.order_date, o.status, p.name, p.price
               FROM orders o JOIN products p ON p.id = o.product_id
               WHERE o.customer_id = ? ORDER BY o.order_date""", (c["id"],)).fetchall()
        lines = [f"order {o['id']} on {o['order_date']}: {o['quantity']} x {o['name']} "
                 f"({o['price']:.0f} dollars each), {o['status']}" for o in orders]
        text = (f"Customer {c['name']} from {c['city']} ({c['tier']} tier) has "
                f"{len(orders)} orders: " + "; ".join(lines) + ".")
        chunks.append(Chunk(f"customer#{c['id']}", text, "customers+orders+products", "entity",
                            {"customer": c["name"], "orders": len(orders)}))
    # products are their own entity
    for p in con.execute("SELECT * FROM products"):
        chunks.append(Chunk(f"products#{p['id']}", _row_to_text("products", dict(p)),
                            "products", "entity", {}))
    return chunks


def chunk_sql_schema(con: sqlite3.Connection) -> list[Chunk]:
    """One chunk per table describing its columns. Used in text-to-SQL setups,
    where the LLM retrieves the schema and then writes a SQL query."""
    chunks = []
    for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        cols = con.execute(f"PRAGMA table_info({table})").fetchall()
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        col_text = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
        chunks.append(Chunk(f"schema#{table}", f"Table {table} has {count} rows with columns: {col_text}.",
                            table, "schema", {"columns": [c["name"] for c in cols]}))
    return chunks


SQL_STRATEGIES = {
    "row": chunk_sql_row,
    "entity": chunk_sql_entity,
    "schema": chunk_sql_schema,
}


def chunk_sql(strategy: str) -> list[Chunk]:
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        return SQL_STRATEGIES[strategy](con)
    finally:
        con.close()
