"""
Phase 7: audit trail + break-testing.

Adds a per-session_id audit log (PRD 6.7 / Section 7's AuditLogEntry) on
top of Phase 6's payment integration. Every search, reasoning step,
confirmation check, and Razorpay API call gets appended with a timestamp,
event_type, payload, and actor — then exported as a flat JSON file a judge
can read without watching the demo video.

New commands in the REPL:
    !export        -- write the current session's audit log to a JSON file
    !breakpayment  -- (from Phase 6) force the next payment attempt to fail

The log also auto-exports when you type 'quit'.

Setup: same as Phase 6 (.env with GEMINI_API_KEYS or GEMINI_API_KEY,
RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET).

Run:
    python phase7_audit_agent.py
"""

import json
import os
import re
import socket
import time
import uuid
from datetime import datetime, timezone

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
VALUE_CAP_INR = int(os.environ.get("VALUE_CAP_INR", 1500))

_raw_keys = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY", "")
API_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]
if not API_KEYS:
    raise SystemExit("No GEMINI_API_KEY or GEMINI_API_KEYS found in .env")
_clients = [genai.Client(api_key=k) for k in API_KEYS]

# Rotation needs to survive across separate `python audit_agent.py` runs —
# an in-memory counter resets to 0 every process start, which meant every
# fresh run kept picking the same (often already-exhausted) first key. A
# tiny local state file fixes that.
_ROTATION_STATE_FILE = ".key_rotation_state"

def _get_next_key_index() -> int:
    try:
        with open(_ROTATION_STATE_FILE) as f:
            index = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        index = 0
    with open(_ROTATION_STATE_FILE, "w") as f:
        f.write(str((index + 1) % len(_clients)))
    return index % len(_clients)


# ---------------------------------------------------------------------------
# Audit trail — matches the PRD's AuditLogEntry shape:
# session_id, timestamp, event_type, payload, actor
# ---------------------------------------------------------------------------

class AuditLog:
    EVENT_TYPES = {"search", "reasoning", "confirmation", "api_call", "error"}

    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.entries = []

    def log(self, event_type: str, payload: dict, actor: str = "system"):
        assert event_type in self.EVENT_TYPES, f"unknown event_type: {event_type}"
        entry = {
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event_type": event_type,
            "actor": actor,
            "payload": payload,
        }
        self.entries.append(entry)
        return entry

    def export(self, path: str = None) -> str:
        path = path or f"audit_log_{self.session_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2, ensure_ascii=False, default=str)
        return path


# ---------------------------------------------------------------------------
# Confirmation gate (unchanged from Phase 5/6)
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
# Tool declarations (same as Phase 6, plus off-topic/ambiguous handling
# instructions per PRD 8's edge cases, needed for break-testing)
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
   exceeds Homestead's \u20b9{VALUE_CAP_INR} standard auto-approval limit. After
   the customer's normal confirmation, you MUST ask a SECOND, clearly
   different question that explicitly names the \u20b9{VALUE_CAP_INR} limit and
   asks them to specifically confirm they want to proceed with this
   higher-value purchase. Only call initiate_payment after that second
   confirmation.
7. Only call initiate_payment after explicit confirmation (and cap
   confirmation if applicable). If it reports the order wasn't confirmed,
   don't argue — calmly ask again.
8. If initiate_payment reports a payment error, explain plainly in one
   sentence (no jargon, no stack traces) and ask if they'd like to try
   again or pick something else. Never pretend it succeeded.
9. If the customer's message is entirely unrelated to shopping for
   Homestead products, gently redirect: mention you can help find something
   in the Homestead catalog and ask what they're looking for. Don't try to
   answer off-topic questions.
10. If a search comes back too broad or the request is too vague to narrow
    meaningfully, ask ONE clarifying question (e.g. who the gift is for, or
    their budget) instead of dumping a long list of options.
11. Never treat a vague, off-topic, or unusual message as a confirmation
    yourself — that determination is made outside of you.
12. Keep replies short.
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
        "if the order exceeded the value cap) as their own separate messages."
    ),
    "parameters": {"type": "object", "properties": {}},
}

TOOLS = [
    search_products_declaration, get_product_declaration, check_stock_declaration,
    start_checkout_declaration, initiate_payment_declaration,
]


class CartTalkSession:
    def __init__(self):
        # previous_interaction_id (used below) is server-side state owned by
        # whichever key/project created it — switching keys mid-conversation
        # would make later calls reference an interaction the new key can't
        # see (404 Not Found). So each session is pinned to ONE client for
        # its whole life; rotation happens across separate sessions instead,
        # persisted to disk so it survives separate script runs.
        key_index = _get_next_key_index()
        self.client_index = key_index
        self.client = _clients[key_index]
        print(f"  [using key #{key_index + 1} of {len(_clients)}]")

        self.previous_interaction_id = None
        self.pending_order = None
        self.force_payment_error = False
        self.audit = AuditLog()
        print(f"  [session_id: {self.audit.session_id}]")

        self.tool_functions = {
            "search_products": self._search_products,
            "get_product": self._get_product,
            "check_stock": self._check_stock,
            "start_checkout": self._start_checkout,
            "initiate_payment": self._initiate_payment,
        }

    # -- audited wrappers around the plain catalog tools ----------------

    def _search_products(self, query: str, max_results: int = 5) -> list:
        results = search_products(query, max_results)
        self.audit.log("search", {
            "raw_query": query,
            "candidate_ids_returned": [p["id"] for p in results],
            "result_count": len(results),
        }, actor="agent")
        return results

    def _get_product(self, product_id: str) -> dict:
        result = get_product(product_id)
        self.audit.log("reasoning", {"tool": "get_product", "args": {"product_id": product_id},
                                      "found": result is not None}, actor="agent")
        return result

    def _check_stock(self, product_id: str) -> dict:
        result = check_stock(product_id)
        self.audit.log("reasoning", {"tool": "check_stock", "args": {"product_id": product_id},
                                      "result": result}, actor="agent")
        return result

    # -- gated tools -----------------------------------------------------

    def _start_checkout(self, product_id: str, quantity: int = 1) -> dict:
        product = get_product(product_id)
        if product is None:
            self.audit.log("error", {"tool": "start_checkout", "reason": f"no such product: {product_id}"}, actor="agent")
            return {"error": f"no such product: {product_id}"}
        total = product["price_inr"] * quantity
        value_cap_triggered = total > VALUE_CAP_INR
        self.pending_order = {
            "product_id": product_id, "name": product["name"], "price_inr": product["price_inr"],
            "quantity": quantity, "total_inr": total,
            "confirmed": False, "value_cap_triggered": value_cap_triggered, "cap_confirmed": False,
        }
        self.audit.log("reasoning", {
            "tool": "start_checkout", "order": self.pending_order,
        }, actor="agent")
        print(f"  [AUDIT] pending order opened: {self.pending_order}")
        return {
            "item": product["name"], "price_inr": product["price_inr"], "quantity": quantity,
            "total_inr": total, "value_cap_triggered": value_cap_triggered,
            "instruction": "Read this back to the customer exactly, then ask them to confirm.",
        }

    def _initiate_payment(self) -> dict:
        if self.pending_order is None:
            self.audit.log("confirmation", {"result": "blocked", "reason": "no pending order"}, actor="system")
            return {"status": "blocked", "reason": "no pending order — nothing to pay for"}
        if not self.pending_order["confirmed"]:
            self.audit.log("confirmation", {"result": "blocked", "reason": "order not confirmed"}, actor="system")
            print("  [AUDIT] initiate_payment BLOCKED — order not confirmed")
            return {"status": "blocked", "reason": "order not yet explicitly confirmed"}
        if self.pending_order["value_cap_triggered"] and not self.pending_order["cap_confirmed"]:
            self.audit.log("confirmation", {"result": "blocked", "reason": "value cap breach not separately confirmed"}, actor="system")
            print("  [AUDIT] initiate_payment BLOCKED — value cap breach not separately confirmed")
            return {"status": "blocked", "reason": f"order exceeds the \u20b9{VALUE_CAP_INR} value cap and needs a separate confirmation"}

        order = self.pending_order
        print(f"  [AUDIT] payment attempt: {order} | forced_error={self.force_payment_error}")

        amount_for_order = -1 if self.force_payment_error else order["total_inr"]
        self.force_payment_error = False

        order_result = create_order(
            amount_inr=amount_for_order,
            receipt=f"carttalk_{order['product_id']}",
            notes={"product_id": order["product_id"], "qty": str(order["quantity"])},
        )
        if not order_result["success"]:
            self.audit.log("error", {"stage": "create_order", "error": order_result["error"]}, actor="system")
            print(f"  [AUDIT] payment FAILED at order creation: {order_result['error']}")
            return {"status": "error", "error": order_result["error"]}

        razorpay_order_id = order_result["order"]["id"]
        self.audit.log("api_call", {
            "endpoint": "orders", "status": "success", "razorpay_order_id": razorpay_order_id,
        }, actor="system")

        link_result = create_payment_link(
            amount_inr=order["total_inr"],
            description=f"CartTalk order: {order['name']} x{order['quantity']}",
            razorpay_order_id=razorpay_order_id,
            notes={"product_id": order["product_id"]},
        )
        if not link_result["success"]:
            self.audit.log("error", {"stage": "create_payment_link", "error": link_result["error"]}, actor="system")
            print(f"  [AUDIT] payment FAILED at link creation: {link_result['error']}")
            return {"status": "error", "error": link_result["error"]}

        payment_link = link_result["link"]["short_url"]
        self.audit.log("api_call", {
            "endpoint": "payment_links", "status": "success",
            "razorpay_order_id": razorpay_order_id, "payment_link": payment_link,
            "order": order,
        }, actor="system")
        print(f"  [AUDIT] payment SUCCEEDED: order_id={razorpay_order_id} link={payment_link}")
        self.pending_order = None
        return {
            "status": "success", "payment_link": payment_link,
            "razorpay_order_id": razorpay_order_id, "order": order,
        }

    # -- main loop -----------------------------------------------------

    def send(self, user_message: str) -> str:
        if user_message.strip() == "!breakpayment":
            self.force_payment_error = True
            return "(debug) Next payment attempt will be deliberately malformed to test error handling."
        if user_message.strip() == "!export":
            path = self.audit.export()
            return f"(debug) Audit log exported to {path}"

        self.audit.log("reasoning", {"message": user_message}, actor="user")

        if self.pending_order is not None:
            order = self.pending_order
            if not order["confirmed"]:
                result, path = check_confirmation(user_message)
                self.audit.log("confirmation", {
                    "stage": "order", "raw_text": user_message, "result": result, "path": path,
                }, actor="system")
                print(f"  [AUDIT] confirmation check on {user_message!r} -> {result} ({path})")
                if result is True:
                    order["confirmed"] = True
                elif result is False:
                    self.pending_order = None
            elif order["value_cap_triggered"] and not order["cap_confirmed"]:
                result, path = check_confirmation(user_message)
                self.audit.log("confirmation", {
                    "stage": "value_cap", "raw_text": user_message, "result": result, "path": path,
                }, actor="system")
                print(f"  [AUDIT] CAP confirmation check on {user_message!r} -> {result} ({path})")
                if result is True:
                    order["cap_confirmed"] = True
                elif result is False:
                    self.pending_order = None

        next_input = user_message
        while True:
            interaction = self._create_with_retry(
                model=MODEL, system_instruction=SYSTEM_INSTRUCTION,
                input=next_input, tools=TOOLS,
                previous_interaction_id=self.previous_interaction_id,
            )
            self.previous_interaction_id = interaction.id

            function_calls = [s for s in interaction.steps if s.type == "function_call"]
            if not function_calls:
                self.audit.log("reasoning", {"reply": interaction.output_text}, actor="agent")
                return interaction.output_text

            function_results = []
            for call in function_calls:
                fn = self.tool_functions.get(call.name)
                args = dict(call.arguments)
                print(f"  [tool call] {call.name}({args})")
                result = fn(**args) if fn else {"error": f"unknown tool {call.name}"}
                function_results.append({
                    "type": "function_result", "name": call.name, "call_id": call.id,
                    "result": [{"type": "text", "text": json.dumps(result)}],
                })
            next_input = function_results

    def _create_with_retry(self, **kwargs):
        def is_transient(e):
            s = str(e) + type(e).__name__
            return any(m in s for m in [
                "429", "too_many_requests", "ConnectionError", "ReadError",
                "APIConnectionError", "timeout", "Timeout", "aborted", "10053", "10054",
            ])

        # No conversation exists yet on ANY key (this is the first call of
        # the session) — safe to try every key, since nothing is pinned to
        # one yet. Whichever key succeeds first owns this session from here
        # on, since its interaction id will only exist there.
        if kwargs.get("previous_interaction_id") is None:
            last_err = None
            for offset in range(len(_clients)):
                idx = (self.client_index + offset) % len(_clients)
                try:
                    result = _clients[idx].interactions.create(**kwargs)
                    self.client_index = idx
                    self.client = _clients[idx]
                    return result
                except Exception as e:
                    last_err = e
                    if not is_transient(e):
                        raise
                    print(f"  (key #{idx + 1} failed, trying next: {e})")
            # every key failed on the first message — fall through to
            # waiting below rather than giving up immediately
        else:
            last_err = None

        # Mid-conversation (or all keys exhausted above): previous_interaction_id
        # ties us to self.client specifically, so retry that same key with backoff.
        for attempt in range(1, 4):
            try:
                return self.client.interactions.create(**kwargs)
            except Exception as e:
                last_err = e
                if not is_transient(e):
                    raise
                match = re.search(r"retry in ([\d.]+)s", str(e))
                wait = float(match.group(1)) + 2 if match else min(10 * attempt, 30)
                print(f"  (transient error, retrying same key in {wait:.0f}s: {e})")
                time.sleep(wait)
        raise last_err


if __name__ == "__main__":
    print(f"CartTalk (Phase 7: audit trail). Value cap \u20b9{VALUE_CAP_INR}.")
    print("Commands: !breakpayment (force a payment error), !export (dump audit log), quit\n")
    session = CartTalkSession()
    try:
        while True:
            user_message = input("YOU: ").strip()
            if user_message.lower() in {"quit", "exit"}:
                break
            reply = session.send(user_message)
            print(f"AGENT: {reply}\n")
    finally:
        path = session.audit.export()
        print(f"\nAudit log exported to {path}")