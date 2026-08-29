"""
Phase 2 setup: load homestead_catalog.json into a local SQLite DB.
Run once (or whenever the catalog JSON changes).

Run:
    python db_setup.py
"""

import json
import sqlite3
from pathlib import Path

CATALOG_PATH = Path("catalog.json")
DB_PATH = Path("homestead.db")


def build_db():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        products = json.load(f)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS products")
    cur.execute("""
        CREATE TABLE products (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price_inr INTEGER NOT NULL,
            description TEXT NOT NULL,
            material TEXT,
            size TEXT,
            color_variants TEXT,   -- JSON array, stored as text (sqlite has no array type)
            stock_qty INTEGER NOT NULL,
            tags TEXT              -- JSON array, stored as text
        )
    """)

    for p in products:
        cur.execute(
            """INSERT INTO products
               (id, name, category, price_inr, description, material, size,
                color_variants, stock_qty, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                p["id"], p["name"], p["category"], p["price_inr"], p["description"],
                p.get("material"), p.get("size"),
                json.dumps(p.get("color_variants", [])),
                p["stock_qty"],
                json.dumps(p.get("tags", [])),
            ),
        )

    conn.commit()
    conn.close()
    print(f"Loaded {len(products)} products into {DB_PATH.resolve()}")


if __name__ == "__main__":
    build_db()