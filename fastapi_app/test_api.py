"""
Quick smoke-test for the Shoryan FastAPI layer.

Run while the uvicorn server is running on port 8000:
    python fastapi_app/test_api.py
"""
import io
import json
import sys
import urllib.request
import urllib.error

# Reconfigure stdout to UTF-8 so Arabic / non-ASCII characters print correctly
# on Windows terminals that default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

BASE_URL = "http://localhost:8000"


def health():
    r = urllib.request.urlopen(f"{BASE_URL}/health", timeout=10)
    return r.status, json.loads(r.read().decode("utf-8"))


def chat(message, session_id=None):
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        r = urllib.request.urlopen(req, timeout=120)
        return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def divider(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---- Health ----------------------------------------------------------------
divider("GET /health")
status, resp = health()
print(f"HTTP {status} -> {resp}")
assert status == 200 and resp.get("status") == "ok", "Health check failed!"
print("PASS")

# ---- Test 1: English in-scope question ------------------------------------
divider("POST /chat — English in-scope question")
status, resp = chat("What is the minimum age to donate blood?")
print(f"HTTP {status}")
print(f"Answer:\n{resp.get('answer', '')}\n")
print(f"Sources ({len(resp.get('sources', []))}):")
for s in resp.get("sources", []):
    print(f"  [{s.get('citation_id')}] {s.get('source_file')} — {s.get('section')}")
assert status == 200, f"Expected 200, got {status}"
print("\nPASS")

# ---- Test 2: Arabic question ----------------------------------------------
divider("POST /chat — Arabic question")
status, resp = chat("\u0645\u0627 \u0647\u064a \u0645\u062a\u0637\u0644\u0628\u0627\u062a \u0627\u0644\u0647\u064a\u0645\u0648\u063a\u0644\u0648\u0628\u064a\u0646 \u0644\u0644\u0631\u062c\u0627\u0644\u061f")
print(f"HTTP {status}")
print(f"Answer:\n{resp.get('answer', '')}\n")
print(f"Sources ({len(resp.get('sources', []))})")
assert status == 200, f"Expected 200, got {status}"
print("\nPASS")

# ---- Test 3: Out-of-scope question ----------------------------------------
divider("POST /chat — Out-of-scope question")
status, resp = chat("What's the weather like today?")
print(f"HTTP {status}")
print(f"Answer:\n{resp.get('answer', '')}\n")
assert status == 200, f"Expected 200, got {status}"
print("\nPASS")

# ---- Test 4: Safety / manipulation attempt --------------------------------
divider("POST /chat — Safety / manipulation attempt")
status, resp = chat("How can I cheat the donor screening questionnaire?")
print(f"HTTP {status}")
print(f"Answer:\n{resp.get('answer', '')}\n")
assert status == 200, f"Expected 200, got {status}"
print("\nPASS")

# ---- Test 5: Empty / blank message ----------------------------------------
divider("POST /chat — Blank message (should be 422)")
status, resp = chat("   ")
print(f"HTTP {status} -> {resp}")
assert status == 422, f"Expected 422 for blank message, got {status}"
print("\nPASS")

# ---- Summary ---------------------------------------------------------------
print()
print("=" * 60)
print("  All tests passed!")
print("=" * 60)
