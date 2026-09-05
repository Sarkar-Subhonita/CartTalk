# CartTalk

A conversational shopping assistant for **Homestead**, a home & kitchen gifts
store — built for the Razorpay AI Buildathon (Track 1: AI Growth & Agentic
Commerce). CartTalk discovers products through natural conversation, narrows
down to a few strong candidates with clear trade-offs, and — only after an
explicit, code-enforced confirmation — generates a real Razorpay payment link.

## What it does

- **Discovery**: describe what you want in plain language; the agent
  searches the catalog and narrows to 2-3 candidates with a short
  explanation of the trade-offs between them (price, material, occasion fit).
- **Follow-ups**: asks like "does it come in another color?" resolve against
  what was already shown, not a fresh search.
- **Confirmation gate**: before any payment action, the agent states the
  exact item, price, quantity, and total, and asks for explicit
  confirmation. This is enforced in code, not just by prompting the model —
  the payment function itself refuses to run without a confirmed flag that
  only a deterministic check on the customer's own words can set.
- **Value cap**: orders above a configurable threshold (default ₹1,500)
  require a second, separately-worded confirmation before payment proceeds.
- **Real payments**: confirmed orders create an actual Razorpay Order and
  Payment Link (test mode).
- **Graceful failure**: out-of-stock items and payment errors are explained
  in plain language, never a crash or a silent failure.
- **Audit trail**: every search, tool call, confirmation check, and payment
  API call is logged per session and exportable as JSON — readable on its
  own, without watching the demo video.

## Project structure

| File | Purpose |
|---|---|
| `homestead_catalog.json` | Seed catalog — 24 products, 2 deliberately out of stock |
| `db_setup.py` | Loads the catalog JSON into a local SQLite DB (`homestead.db`) |
| `catalog_tools.py` | `search_products`, `get_product`, `check_stock` — pure catalog logic, no LLM |
| `razorpay_client.py` | Razorpay Orders + Payment Links API calls, with error handling |
| `audit_agent.py` | **Main entry point.** The full conversational agent: discovery, narrowing, confirmation gate, payments, and audit logging |
| `proofscript.py` | Standalone script proving the Razorpay Orders → Payment Links flow works independently of the agent (see PRD requirement 6.8) |
| `test_catalog_tools.py` | Standalone tests for the catalog functions, no LLM involved |
| `audit_log_happy_path.json` | Sample audit log from a full successful purchase |
| `audit_log_out_of_stock.json` | Sample audit log from a session that hit an out-of-stock item |

## Setup

**Requirements**: Python 3.10+, a Gemini API key, and Razorpay test-mode keys.

1. Clone the repo and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # Mac/Linux
   ```

2. Install dependencies:
   ```
   pip install google-genai requests python-dotenv
   ```

3. Create a `.env` file in the project root (see `.env.example`):
   ```
   GEMINI_API_KEYS=your-gemini-key-1,your-gemini-key-2
   RAZORPAY_KEY_ID=rzp_test_xxxxxxxx
   RAZORPAY_KEY_SECRET=your-razorpay-test-secret
   VALUE_CAP_INR=1500
   ```
   - `GEMINI_API_KEYS` accepts a comma-separated list (recommended, for
     free-tier quota headroom) or a single key. Get keys at
     [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
   - Razorpay test keys: Dashboard → Settings → API Keys, with the
     dashboard toggled to **Test Mode**.
   - `VALUE_CAP_INR` is optional (defaults to 1500).

4. Build the local product database (run once, or whenever the catalog
   JSON changes):
   ```
   python db_setup.py
   ```

## Running it

```
python audit_agent.py
```

Then just chat with it like a customer — e.g. *"I need a gift for a
friend's housewarming, budget around 2000 rupees."*

**Debug commands** (typed as a message, not sent to the model):
- `!breakpayment` — forces the *next* payment attempt to fail with a real
  malformed-request error from Razorpay, to see the error-handling path.
- `!export` — writes the current session's audit log to
  `audit_log_<session_id>.json` immediately (it also auto-exports when you
  type `quit`, or on a crash).

## Two demo paths

1. **Happy path**: search → narrow → pick an item → confirm → real payment
   link.
2. **Out-of-stock path**: a natural query (e.g. *"something nice for a
   housewarming"*) surfaces one of the two deliberately-out-of-stock SKUs
   as a candidate; the agent states it's unavailable and offers an
   in-stock alternative instead of silently substituting or failing.

## Verifying the payment flow independently

Per the track's requirement that the payment integration work standalone,
`proofscript.py` calls the Orders API and Payment Links API directly, with
no agent or LLM involved:
```
python proofscript.py
```

## Notes

- All payments run in Razorpay **test mode** — no real money moves.
- The confirmation gate is deliberately conservative: any message that
  doesn't clearly match a small whitelist of affirmatives (or negatives)
  is treated as **not confirmed**, and the agent just asks again. This is
  intentional — it should never be possible to talk the agent into
  skipping confirmation.
- Scope is intentionally narrow to what the two demo paths need (per the
  track's stated non-goals) rather than general-purpose robustness.