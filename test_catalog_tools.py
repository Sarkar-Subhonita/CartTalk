"""
Phase 2 exit check: call search_products / get_product / check_stock
directly (no LLM, no agent) against 6 sample queries and eyeball the results.

Run:
    python test_catalog_tools.py
"""

from catalog_tools import search_products, get_product, check_stock

SAMPLE_QUERIES = [
    "gift for a housewarming under 2000",     # should surface the out-of-stock tawa/canisters as candidates
    "something for a coffee lover",
    "budget gift under 700",
    "kitchen storage under 1200",
    "dinner plates for a new apartment",
    "something nice for a housewarming",       # deliberately vague — no strong tokens
]

for q in SAMPLE_QUERIES:
    print(f"\nQUERY: {q!r}")
    results = search_products(q)
    if not results:
        print("  (no results)")
    for p in results:
        stock_note = "OUT OF STOCK" if p["stock_qty"] == 0 else f"stock={p['stock_qty']}"
        print(f"  {p['id']} | {p['name']} | ₹{p['price_inr']} | {stock_note}")

print("\n--- get_product / check_stock checks ---")
print("get_product('HS-009') ->", get_product("HS-009"))   # cast iron tawa, out of stock
print("check_stock('HS-009') ->", check_stock("HS-009"))
print("check_stock('HS-001') ->", check_stock("HS-001"))
print("check_stock('HS-999') ->", check_stock("HS-999"))   # doesn't exist