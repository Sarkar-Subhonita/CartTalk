"""
Razorpay Orders + Payment Links calls — the same flow proven standalone in
Phase 1's proof script, but now with real error handling: every function
here returns a {"success": True/False, ...} dict instead of raising, since
an uncaught exception inside a tool function would crash the whole agent
(exactly the "raw crash" PRD 6.6 says must not happen).
"""

import os
import requests
from requests.auth import HTTPBasicAuth

RAZORPAY_KEY_ID = os.environ.get("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET")
BASE_URL = "https://api.razorpay.com/v1"


def _auth():
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        raise RuntimeError(
            "Missing RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET — check your .env file."
        )
    return HTTPBasicAuth(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)


def create_order(amount_inr, receipt: str, notes: dict | None = None) -> dict:
    """Create a Razorpay Order. Returns {'success': True, 'order': {...}} or
    {'success': False, 'error': '...'} — never raises for a normal API-level
    failure (auth/setup problems still raise, since those aren't recoverable
    mid-conversation)."""
    try:
        resp = requests.post(
            f"{BASE_URL}/orders",
            auth=_auth(),
            json={
                "amount": int(round(amount_inr * 100)),  # rupees -> paise
                "currency": "INR",
                "receipt": receipt,
                "notes": notes or {},
            },
            timeout=15,
        )
        resp.raise_for_status()
        return {"success": True, "order": resp.json()}
    except requests.exceptions.HTTPError as e:
        # Razorpay returns a JSON body with the actual reason even on 4xx/5xx
        try:
            detail = e.response.json().get("error", {}).get("description", str(e))
        except Exception:
            detail = str(e)
        return {"success": False, "error": f"Razorpay order creation failed: {detail}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Network error creating order: {e}"}


def create_payment_link(amount_inr, description: str, razorpay_order_id: str,
                         notes: dict | None = None) -> dict:
    """Create a Payment Link. Same success/error dict shape as create_order."""
    try:
        resp = requests.post(
            f"{BASE_URL}/payment_links",
            auth=_auth(),
            json={
                "amount": int(round(amount_inr * 100)),
                "currency": "INR",
                "description": description,
                "reminder_enable": False,
                "notes": {**(notes or {}), "internal_order_id": razorpay_order_id},
            },
            timeout=15,
        )
        resp.raise_for_status()
        return {"success": True, "link": resp.json()}
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("error", {}).get("description", str(e))
        except Exception:
            detail = str(e)
        return {"success": False, "error": f"Razorpay payment link creation failed: {detail}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Network error creating payment link: {e}"}