"""
Phase 3: get Gemini to reliably call search_products from natural language.
No checkout, no narrowing/trade-off logic yet — this phase is ONLY about
tool-invocation reliability: does it call the right tool, with sensible
arguments, across a handful of different natural-language intents?

Setup:
    pip install google-genai
    $env:GEMINI_API_KEY = "your-key-from-aistudio.google.com"   (PowerShell)

Run:
    python phase3_agent.py
"""

import socket

# Workaround: some networks have a broken/blackholed IPv6 path where the TCP
# handshake succeeds but data transfer hangs forever. Forcing DNS resolution
# to IPv4-only avoids it without touching system network settings.
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_only_getaddrinfo

from google import genai
from catalog_tools import search_products

MODEL = "gemini-3.6-flash"  # per Google's own 404 error message recommending this
# over the now-deprecated gemini-2.5-flash

search_products_declaration = {
    "type": "function",
    "name": "search_products",
    "description": (
        "Searches the Homestead home & kitchen product catalog by a natural-language "
        "query. Understands budget phrases like 'under 2000' or 'less than 1k'. "
        "Returns matching products ranked by relevance, including out-of-stock items "
        "(stock status is a field on each result, not something this search filters out)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user's natural-language product request, e.g. 'a gift for a housewarming under 2000'.",
            },
            "max_results": {
                "type": "integer",
                "description": "Max number of products to return. Defaults to 5 if omitted.",
            },
        },
        "required": ["query"],
    },
}

client = genai.Client()


def _create_with_retry(max_attempts: int = 5, **kwargs):
    """Gemini occasionally returns a transient 'high demand' server error.
    Retry with a short exponential backoff before giving up — this matters
    more than it looks, since you don't want the demo recording interrupted
    by a temporary overload."""
    import time
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return client.interactions.create(**kwargs)
        except Exception as e:
            last_err = e
            is_rate_limit = "429" in str(e) or "too_many_requests" in str(e)
            if not is_rate_limit:
                raise  # permanent error (bad model name, auth, etc) — retrying won't help
            if attempt < max_attempts:
                wait = min(10 * attempt, 30)  # 10s, 20s, 30s, 30s... covers rate-limit resets (~50s)
                print(f"  (transient error, retrying in {wait}s: {e})")
                time.sleep(wait)
    raise last_err


def run_query(user_message: str):
    print(f"\nUSER: {user_message}")
    interaction = _create_with_retry(
        model=MODEL,
        input=user_message,
        tools=[search_products_declaration],
    )

    fc_steps = [s for s in interaction.steps if s.type == "function_call"]
    if not fc_steps:
        print("  (!) Gemini did NOT call a tool for this message.")
        print("  Model said instead:", interaction.output_text)
        return

    for fc_step in fc_steps:
        print(f"  TOOL CALL: {fc_step.name}({dict(fc_step.arguments)})")
        if fc_step.name == "search_products":
            results = search_products(**dict(fc_step.arguments))
            print(f"  -> {len(results)} result(s):")
            for p in results[:5]:
                stock_note = "OUT OF STOCK" if p["stock_qty"] == 0 else f"stock={p['stock_qty']}"
                print(f"     {p['id']} | {p['name']} | \u20b9{p['price_inr']} | {stock_note}")
        else:
            print(f"  (!) Unexpected tool name: {fc_step.name}")


if __name__ == "__main__":
    # 4 different natural-language intents — this is the Phase 3 exit check.
    TEST_MESSAGES = [
        "I need a gift for a friend's housewarming, budget around 2000 rupees",
        "do you have anything for someone who loves coffee?",
        "looking for something for the kitchen, nothing too expensive",
        "what dinnerware do you have?",
    ]
    for msg in TEST_MESSAGES:
        run_query(msg)