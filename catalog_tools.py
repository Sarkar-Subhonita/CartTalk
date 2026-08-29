"""
Phase 2: the three catalog tool functions CartTalk's agent will eventually
call via tool-calling. Kept dependency-free (stdlib only) so Phase 3 can
wire them into Gemini's function-calling without any adapter layer.
"""

import json
import re
import sqlite3
from pathlib import Path

DB_PATH = Path("homestead.db")


def _row_to_dict(row, columns):
    d = dict(zip(columns, row))
    d["color_variants"] = json.loads(d["color_variants"] or "[]")
    d["tags"] = json.loads(d["tags"] or "[]")
    return d


def _extract_budget(query: str):
    """Pull a rupee ceiling out of phrases like 'under 2000', 'under ₹1500',
    'less than 2k', 'budget of 1000'. Returns None if no budget is mentioned.
    """
    q = query.lower()
    match = re.search(
        r"(?:under|below|less than|budget(?: of)?)\s*(?:₹|rs\.?|inr)?\s*(\d+)(k)?",
        q,
    )
    if not match:
        return None
    amount = int(match.group(1))
    if match.group(2):  # "2k" -> 2000
        amount *= 1000
    return amount


def search_products(query: str, max_results: int = 5) -> list[dict]:
    """
    Keyword + tag matching search (per PRD open question #2 — the lower-risk
    choice for an 8-day solo build over embedding similarity).

    Scores every product — in stock or not — by how many query tokens appear
    in its tags/category/name/description, tags weighted highest since
    they're the most deliberate signal. Applies a budget ceiling if the
    query mentions one.

    Deliberately does NOT filter out out-of-stock items: per PRD 6.6, the
    seeded out-of-stock SKU has to be reachable through the agent's normal
    search/narrowing logic, not an unnatural direct query. Stock is just
    another field on the returned dict — it's the agent's job (via
    check_stock, later phases) to notice and handle it gracefully.

    Returns a list of product dicts, best match first, capped at max_results.
    """
    budget = _extract_budget(query)
    tokens = [t for t in re.findall(r"[a-z]+", query.lower()) if len(t) > 2]

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM products")
    columns = [c[0] for c in cur.description]
    rows = cur.fetchall()
    conn.close()

    scored = []
    for row in rows:
        product = _row_to_dict(row, columns)
        if budget is not None and product["price_inr"] > budget:
            continue

        haystack_tags = " ".join(product["tags"]).lower()
        haystack_name_desc = f"{product['name']} {product['description']} {product['category']}".lower()

        score = 0
        for tok in tokens:
            if tok in haystack_tags:
                score += 3
            if tok in haystack_name_desc:
                score += 1

        # If the query had no usable tokens at all (e.g. just a budget),
        # still include everything within budget rather than returning nothing.
        if score > 0 or not tokens:
            scored.append((score, product))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:max_results]]


def get_product(product_id: str) -> dict | None:
    """Fetch full details for one product by id, regardless of stock status."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM products WHERE id = ?", (product_id,))
    columns = [c[0] for c in cur.description]
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return _row_to_dict(row, columns)


def check_stock(product_id: str) -> dict:
    """Small, agent-friendly shape rather than the full product record —
    this is what the confirmation/failure-handling logic in later phases
    will call right before committing to an item."""
    product = get_product(product_id)
    if product is None:
        return {"product_id": product_id, "in_stock": False, "stock_qty": 0, "error": "not found"}
    return {
        "product_id": product_id,
        "in_stock": product["stock_qty"] > 0,
        "stock_qty": product["stock_qty"],
    }