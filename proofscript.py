"""
Phase 1 exit check for CartTalk:
Paste a hardcoded order in, get back a real Razorpay test-mode payment link.
No agent, no catalog, no LLM involved yet — just proving the two API calls work.

Setup:
  pip install requests
  export RAZORPAY_KEY_ID="rzp_test_xxxxxxxxxxxx"
  export RAZORPAY_KEY_SECRET="your_test_key_secret"

Run:
  python razorpay_proof_script.py
"""

import os
import requests
from requests.auth import HTTPBasicAuth

RAZORPAY_KEY_ID = os.environ.get("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET")
BASE_URL = "https://api.razorpay.com/v1"

if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
    raise SystemExit(
        "Missing RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET env vars.\n"
        "Get test-mode keys from Razorpay Dashboard -> Settings -> API Keys "
        "(make sure the dashboard is toggled to Test Mode)."
    )

AUTH = HTTPBasicAuth(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)


def create_order(amount_inr: float, receipt: str, notes: dict | None = None) -> dict:
    """Create a Razorpay Order. Amount must be sent in paise, not rupees."""
    resp = requests.post(
        f"{BASE_URL}/orders",
        auth=AUTH,
        json={
            "amount": int(round(amount_inr * 100)),
            "currency": "INR",
            "receipt": receipt,
            "notes": notes or {},
        },
    )
    resp.raise_for_status()
    return resp.json()


def create_payment_link(amount_inr: float, description: str, razorpay_order_id: str,
                         notes: dict | None = None) -> dict:
    """Create a Payment Link tied back to the Order created above via notes.
    (Payment Links generates its own internal order; we cross-reference our
    Orders-API order id in notes so the audit trail can join the two later.)"""
    resp = requests.post(
        f"{BASE_URL}/payment_links",
        auth=AUTH,
        json={
            "amount": int(round(amount_inr * 100)),
            "currency": "INR",
            "description": description,
            "reminder_enable": False,
            "notes": {**(notes or {}), "internal_order_id": razorpay_order_id},
        },
    )
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    # --- Hardcoded stand-in order (Phase 2 will replace this with real
    # catalog + agent-narrowed selections) ---
    hardcoded_order = {
        "item_name": "Handwoven Jute Table Runner",
        "product_id": "HS-001",
        "quantity": 1,
        "price_inr": 799,
    }
    total_inr = hardcoded_order["price_inr"] * hardcoded_order["quantity"]

    print(f"Creating order for {hardcoded_order['item_name']} — ₹{total_inr}...")
    order = create_order(
        amount_inr=total_inr,
        receipt=f"carttalk_test_{hardcoded_order['product_id']}",
        notes={"product_id": hardcoded_order["product_id"], "qty": str(hardcoded_order["quantity"])},
    )
    print("  order id:", order["id"], "| status:", order["status"])

    print("Creating payment link...")
    link = create_payment_link(
        amount_inr=total_inr,
        description=f"CartTalk order: {hardcoded_order['item_name']} x{hardcoded_order['quantity']}",
        razorpay_order_id=order["id"],
        notes={"product_id": hardcoded_order["product_id"]},
    )
    print("  payment link:", link["short_url"])
    print("\nExit check passed if the link above opens a real Razorpay test-mode checkout page.")