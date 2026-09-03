"""
Phase 6: payment integration + guardrails.

Builds on Phase 5's confirmation gate:
- Confirmed orders now actually call Razorpay (Orders API -> Payment Links
  API, same flow proven standalone in Phase 1) instead of a stub.
- A config-driven value cap (PRD 6.4): orders above it require a SECOND,
  distinct confirmation step before initiate_payment will proceed — not
  just repeating the same question.
- A simulated Razorpay API error path (PRD 6.6): type '!breakpayment' right
  before confirming an order to force the next payment attempt to fail with
  a real malformed-request error from Razorpay, and see it get caught and
  explained conversationally instead of crashing.

Setup: same as Phase 5 (.env with GEMINI_API_KEY, RAZORPAY_KEY_ID,
RAZORPAY_KEY_SECRET).

Run:
    python phase6_payment_agent.py
"""

import json
import os
import re
import socket
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Workaround: some networks have a broken/blackholed IPv6 path where the TCP
# handshake succeeds but data transfer hangs forever. Forcing DNS resolution
# to IPv4-only avoids it without touching system network settings.
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_only_getaddrinfo

from google import genai
from catalog_tools import search_products, get_product, check_stock
from rzp_client import create_order, create_payment_link

MODEL = "gemini-3.6-flash"

# --- Multi-key rotation: reads a comma-separated GEMINI_API_KEYS from .env
# (falls back to single GEMINI_API_KEY if that's all you have). The moment
# one key gets rate-limited, we immediately try the next key instead of
# sitting through a 45-60s wait — this has been the single biggest source
# of lost debugging time today, worth fixing properly rather than routing
# around it key-by-key each time it happens. ---
_raw_keys = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY", "")
API_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]
if not API_KEYS:
    raise SystemExit("No GEMINI_API_KEY or GEMINI_API_KEYS found in .env")
_clients = [genai.Client(api_key=k) for k in API_KEYS]
_current_key_index = 0

# --- Value cap (PRD 6.4): config-driven, not hardcoded inline. Override via
# .env if you want a different threshold; defaults chosen to plausibly
# trigger within a ~2000-2500 gift-budget demo per the build plan. ---
VALUE_CAP_INR = int(os.environ.get("VALUE_CAP_INR", 1500))


# ---------------------------------------------------------------------------
# The confirmation gate (from Phase 5, unchanged)
# ---------------------------------------------------------------------------

_CONFIRM_WHITELIST = [
    r"^y$", r"^ya$", r"^yes\b", r"^yeah\b", r"^yep\b", r"^yup\b",
    r"^confirm(ed)?$", r"^ok(ay)?, confirm(ed)?$",
    r"^go ahead$", r"^proceed$", r"^place (the )?order$", r"^book it$",
    r"^do it$", r"^that works,? (confirm|go ahead|proceed)$",
]
_DECLINE_WHITELIST = [
    r"^n$", r"^no\b", r"^nope\b", r"^nah\b", r"^cancel\b", r"^wait\b",
    r"^stop\b", r"^don'?t\b", r"^actually\b", r"^hold on\b",
]


def check_confirmation(raw_text: str):
    text = raw_text.strip().lower()
    for pattern in _CONFIRM_WHITELIST:
        if re.match(pattern, text):
            return True, f"whitelist_confirm:{pattern}"
    for pattern in _DECLINE_WHITELIST:
        if re.match(pattern, text):
            return False, f"whitelist_decline:{pattern}"
    return None, "ambiguous_defaulted_false"


# ---------------------------------------------------------------------------
# Tool declarations
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = f"""
You are CartTalk, a conversational shopping assistant for Homestead, a home
& kitchen gifts store.

Behavior:
1. When the customer describes what they want, call search_products.
2. From the results, narrow down to 2-3 strong candidates and briefly
   explain the trade-offs (price, material, occasion/budget fit). Be
   concise and conversational.
3. If a candidate is out of stock, say so clearly and suggest an in-stock
   alternative rather than presenting it as a live option.
4. Handle natural follow-ups by resolving against candidates you already
   narrowed to (get_product / check_stock), not a fresh search.
5. When the customer wants to buy something, call start_checkout. Then
   state the EXACT item, price, quantity, and total from its result, and
   ask the customer to confirm. Do not call initiate_payment this turn.
6. If start_checkout's result shows value_cap_triggered=true, this order
   exceeds Homestead's ₹{VALUE_CAP_INR} standard auto-approval limit. After
   the customer's normal confirmation, you MUST ask a SECOND, clearly
   different question that explicitly names the ₹{VALUE_CAP_INR} limit and
   asks them to specifically confirm they want to proceed with this
   higher-value purchase. Do not just repeat "please confirm" — make it
   obviously a distinct, extra step. Only call initiate_payment after that
   second confirmation.
7. Only call initiate_payment after the customer has explicitly confirmed
   (and, if applicable, given the second cap confirmation too). If it
   reports the order wasn't confirmed, don't argue — calmly ask again.
8. If initiate_payment reports a payment error, explain plainly what
   happened in one sentence (no technical jargon, no stack traces) and ask
   if they'd like to try again or pick something else. Never pretend it
   succeeded.
9. Never treat a vague, off-topic, or unusual message as a confirmation
   yourself — that determination is made outside of you.
10. Keep replies short.
"""

search_products_declaration = {
    "type": "function", "name": "search_products",
    "description": (
        "Searches the Homestead catalog by natural-language query. Understands "
        "budget phrases like 'under 2000'. Returns matches including out-of-stock items."
    ),
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
        "required": ["query"],
    },
}
get_product_declaration = {
    "type": "function", "name": "get_product",
    "description": "Fetch full details for a single product by its id, regardless of stock status.",
    "parameters": {"type": "object", "properties": {"product_id": {"type": "string"}}, "required": ["product_id"]},
}
check_stock_declaration = {
    "type": "function", "name": "check_stock",
    "description": "Check current stock level for a single product id.",
    "parameters": {"type": "object", "properties": {"product_id": {"type": "string"}}, "required": ["product_id"]},
}
start_checkout_declaration = {
    "type": "function", "name": "start_checkout",
    "description": (
        "Call when the customer wants to buy a specific item. Returns the exact order "
        "summary (item, price, quantity, total, value_cap_triggered) to read back verbatim "
        "before asking them to confirm. Does NOT charge anything or count as confirmation."
    ),
    "parameters": {
        "type": "object",
        "properties": {"product_id": {"type": "string"}, "quantity": {"type": "integer", "description": "Defaults to 1."}},
        "required": ["product_id"],
    },
}
initiate_payment_declaration = {
    "type": "function", "name": "initiate_payment",
    "description": (
        "Attempts to create a real Razorpay payment link for the pending order. Only "
        "succeeds if the customer has explicitly confirmed (and given a second confirmation "
        "if the order exceeded the value cap) as their own separate messages — calling this "
        "otherwise is blocked regardless of what you believe the customer meant."
    ),
    "parameters": {"type": "object", "properties": {}},
}

TOOLS = [
    search_products_declaration, get_product_declaration, check_stock_declaration,
    start_checkout_declaration, initiate_payment_declaration,
]


class CartTalkSession:
    def __init__(self):
        self.previous_interaction_id = None
        self.pending_order = None
        self.force_payment_error = False  # set by the '!breakpayment' debug command

        self.tool_functions = {
            "search_products": search_products,
            "get_product": get_product,
            "check_stock": check_stock,
            "start_checkout": self._start_checkout,
            "initiate_payment": self._initiate_payment,
        }

    # -- gated tools ---------------------------------------------------

    def _start_checkout(self, product_id: str, quantity: int = 1) -> dict:
        product = get_product(product_id)
        if product is None:
            return {"error": f"no such product: {product_id}"}
        total = product["price_inr"] * quantity
        value_cap_triggered = total > VALUE_CAP_INR
        self.pending_order = {
            "product_id": product_id,
            "name": product["name"],
            "price_inr": product["price_inr"],
            "quantity": quantity,
            "total_inr": total,
            "confirmed": False,
            "value_cap_triggered": value_cap_triggered,
            "cap_confirmed": False,
        }
        print(f"  [AUDIT] pending order opened: {self.pending_order}")
        return {
            "item": product["name"],
            "price_inr": product["price_inr"],
            "quantity": quantity,
            "total_inr": total,
            "value_cap_triggered": value_cap_triggered,
            "instruction": "Read this back to the customer exactly, then ask them to confirm.",
        }

    def _initiate_payment(self) -> dict:
        # THE GATE. Pure code, no arguments the model could use to talk its
        # way past any of these checks.
        if self.pending_order is None:
            return {"status": "blocked", "reason": "no pending order — nothing to pay for"}
        if not self.pending_order["confirmed"]:
            print("  [AUDIT] initiate_payment BLOCKED — order not confirmed")
            return {"status": "blocked", "reason": "order not yet explicitly confirmed"}
        if self.pending_order["value_cap_triggered"] and not self.pending_order["cap_confirmed"]:
            print("  [AUDIT] initiate_payment BLOCKED — value cap breach not separately confirmed")
            return {
                "status": "blocked",
                "reason": f"order exceeds the \u20b9{VALUE_CAP_INR} value cap and needs a separate confirmation",
            }

        order = self.pending_order
        print(f"  [AUDIT] payment attempt: {order} | forced_error={self.force_payment_error}")

        # Deliberately malformed amount if the debug flag is set — this is
        # PRD 6.6's required simulated Razorpay API error.
        amount_for_order = -1 if self.force_payment_error else order["total_inr"]
        self.force_payment_error = False  # one-shot

        order_result = create_order(
            amount_inr=amount_for_order,
            receipt=f"carttalk_{order['product_id']}",
            notes={"product_id": order["product_id"], "qty": str(order["quantity"])},
        )
        if not order_result["success"]:
            print(f"  [AUDIT] payment FAILED at order creation: {order_result['error']}")
            return {"status": "error", "error": order_result["error"]}

        razorpay_order_id = order_result["order"]["id"]
        link_result = create_payment_link(
            amount_inr=order["total_inr"],
            description=f"CartTalk order: {order['name']} x{order['quantity']}",
            razorpay_order_id=razorpay_order_id,
            notes={"product_id": order["product_id"]},
        )
        if not link_result["success"]:
            print(f"  [AUDIT] payment FAILED at link creation: {link_result['error']}")
            return {"status": "error", "error": link_result["error"]}

        payment_link = link_result["link"]["short_url"]
        print(f"  [AUDIT] payment SUCCEEDED: order_id={razorpay_order_id} link={payment_link}")
        self.pending_order = None  # clear — prevents re-triggering payment on this order
        return {
            "status": "success",
            "payment_link": payment_link,
            "razorpay_order_id": razorpay_order_id,
            "order": order,
        }

    # -- main loop -----------------------------------------------------

    def send(self, user_message: str) -> str:
        # Debug command, handled entirely outside the model.
        if user_message.strip() == "!breakpayment":
            self.force_payment_error = True
            return "(debug) Next payment attempt will be deliberately malformed to test error handling."

        if self.pending_order is not None:
            order = self.pending_order
            if not order["confirmed"]:
                result, path = check_confirmation(user_message)
                print(f"  [AUDIT] confirmation check on {user_message!r} -> {result} ({path})")
                if result is True:
                    order["confirmed"] = True
                elif result is False:
                    self.pending_order = None
            elif order["value_cap_triggered"] and not order["cap_confirmed"]:
                result, path = check_confirmation(user_message)
                print(f"  [AUDIT] CAP confirmation check on {user_message!r} -> {result} ({path})")
                if result is True:
                    order["cap_confirmed"] = True
                elif result is False:
                    self.pending_order = None

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
                fn = self.tool_functions.get(call.name)
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

    def _create_with_retry(self, **kwargs):
        global _current_key_index
        last_err = None
        # First pass: try every key once, immediately, no waiting.
        for _ in range(len(_clients)):
            try:
                return _clients[_current_key_index].interactions.create(**kwargs)
            except Exception as e:
                last_err = e
                is_rate_limit = "429" in str(e) or "too_many_requests" in str(e)
                if not is_rate_limit:
                    raise
                print(f"  (key #{_current_key_index + 1} rate limited, switching key)")
                _current_key_index = (_current_key_index + 1) % len(_clients)

        # All keys are rate-limited — now actually wait, using Google's
        # suggested time if it gave one.
        for attempt in range(1, 3):
            match = re.search(r"retry in ([\d.]+)s", str(last_err))
            wait = float(match.group(1)) + 2 if match else 30
            print(f"  (all keys rate limited, waiting {wait:.0f}s)")
            time.sleep(wait)
            try:
                return _clients[_current_key_index].interactions.create(**kwargs)
            except Exception as e:
                last_err = e
                if not ("429" in str(e) or "too_many_requests" in str(e)):
                    raise
        raise last_err


if __name__ == "__main__":
    print(f"CartTalk (Phase 6: real payments, value cap \u20b9{VALUE_CAP_INR}). "
          f"Type '!breakpayment' before confirming to test error handling. Type 'quit' to exit.\n")
    session = CartTalkSession()
    while True:
        user_message = input("YOU: ").strip()
        if user_message.lower() in {"quit", "exit"}:
            break
        reply = session.send(user_message)
        print(f"AGENT: {reply}\n")