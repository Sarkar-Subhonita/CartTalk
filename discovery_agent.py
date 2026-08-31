"""
Phase 4: the full discover -> narrow -> explain loop, conversational,
with follow-up handling within a session (e.g. "does it come in another
color?" resolving against the already-narrowed candidates).

Still no checkout/payment logic — that's Phase 5 (confirmation gate) and
Phase 6 (payment integration). This phase is purely: can the agent hold a
natural multi-turn conversation that narrows to 2-3 candidates and explains
the trade-offs between them?

Setup: same as Phase 3 (google-genai installed, GEMINI_API_KEY set).

Run:
    python phase4_agent.py
Then just type like you're a customer. Type 'quit' to exit.
"""

import json
import socket
import time

# Workaround: some networks have a broken/blackholed IPv6 path where the TCP
# handshake succeeds but data transfer hangs forever. Forcing DNS resolution
# to IPv4-only avoids it without touching system network settings.
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_only_getaddrinfo

from google import genai
from catalog_tools import search_products, get_product, check_stock

MODEL = "gemini-3.6-flash"

SYSTEM_INSTRUCTION = """
You are CartTalk, a conversational shopping assistant for Homestead, a home
& kitchen gifts store.

You are DISCOVERY ONLY — you cannot take payment or place an order yet.
Never claim to place an order, ask for payment details, or tell the
customer to go complete checkout "on the website" or anywhere else — there
is no separate website. When the customer is ready to buy, simply say that
checkout isn't available in this conversation yet.

Behavior:
1. When the customer describes what they want, call search_products.
2. From the results, narrow down to 2-3 strong candidates — don't just list
   everything that matched.
3. Briefly explain the trade-offs between the 2-3 candidates (price,
   material, what occasion or budget each suits best). Be concise and
   conversational, like a helpful shop assistant texting back — not a spec
   sheet or a formal report.
4. If a candidate is out of stock, say so clearly and don't present it as a
   live option alongside in-stock ones — but you can still describe it if
   the customer seems drawn to it, then suggest the closest in-stock
   alternative.
5. Handle natural follow-ups (e.g. "does it come in another color?", "which
   one's cheaper?") by resolving against the candidates you already
   narrowed to, using get_product or check_stock as needed. Only run a new
   search_products call if the customer's request clearly changes what
   they're looking for.
6. Keep replies short.
"""

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
            "query": {"type": "string", "description": "The user's natural-language product request."},
            "max_results": {"type": "integer", "description": "Max results to return. Defaults to 5."},
        },
        "required": ["query"],
    },
}

get_product_declaration = {
    "type": "function",
    "name": "get_product",
    "description": "Fetch full details for a single product by its id, regardless of stock status.",
    "parameters": {
        "type": "object",
        "properties": {"product_id": {"type": "string"}},
        "required": ["product_id"],
    },
}

check_stock_declaration = {
    "type": "function",
    "name": "check_stock",
    "description": "Check current stock level for a single product id.",
    "parameters": {
        "type": "object",
        "properties": {"product_id": {"type": "string"}},
        "required": ["product_id"],
    },
}

TOOLS = [search_products_declaration, get_product_declaration, check_stock_declaration]
TOOL_FUNCTIONS = {
    "search_products": search_products,
    "get_product": get_product,
    "check_stock": check_stock,
}

client = genai.Client()


class CartTalkSession:
    """One conversational session (mirrors the PRD's session_id concept).
    Wraps Gemini's server-side previous_interaction_id chaining and the
    function-call round trip, so callers just send messages and get text
    back without juggling interaction IDs or the tool loop themselves."""

    def __init__(self):
        self.previous_interaction_id = None

    def send(self, user_message: str) -> str:
        next_input = user_message
        while True:
            interaction = self._create_with_retry(
                model=MODEL,
                system_instruction=SYSTEM_INSTRUCTION,
                input=next_input,
                tools=TOOLS,
                previous_interaction_id=self.previous_interaction_id,
            )
            self.previous_interaction_id = interaction.id

            function_calls = [s for s in interaction.steps if s.type == "function_call"]
            if not function_calls:
                return interaction.output_text

            function_results = []
            for call in function_calls:
                fn = TOOL_FUNCTIONS.get(call.name)
                args = dict(call.arguments)
                print(f"  [tool call] {call.name}({args})")
                result = fn(**args) if fn else {"error": f"unknown tool {call.name}"}
                function_results.append({
                    "type": "function_result",
                    "name": call.name,
                    "call_id": call.id,
                    "result": [{"type": "text", "text": json.dumps(result)}],
                })
            next_input = function_results

    def _create_with_retry(self, max_attempts: int = 5, **kwargs):
        last_err = None
        for attempt in range(1, max_attempts + 1):
            try:
                return client.interactions.create(**kwargs)
            except Exception as e:
                last_err = e
                is_rate_limit = "429" in str(e) or "too_many_requests" in str(e)
                if not is_rate_limit:
                    raise  # permanent error — retrying won't help
                if attempt < max_attempts:
                    wait = min(10 * attempt, 30)
                    print(f"  (rate limited, retrying in {wait}s)")
                    time.sleep(wait)
        raise last_err


if __name__ == "__main__":
    print("CartTalk (discovery only, no checkout yet). Type 'quit' to exit.\n")
    session = CartTalkSession()
    while True:
        user_message = input("YOU: ").strip()
        if user_message.lower() in {"quit", "exit"}:
            break
        reply = session.send(user_message)
        print(f"AGENT: {reply}\n")