"""Quick smoke test — verifies all critical endpoints work correctly."""
import httpx, json, sys

BASE = "http://localhost:7861"
PASS = []
FAIL = []

def check(name, condition, detail=""):
    if condition:
        print(f"[PASS] {name}")
        PASS.append(name)
    else:
        print(f"[FAIL] {name} — {detail}")
        FAIL.append(name)

# ── Test 1: /reset with empty body (pre-validation gate) ─────────────────────
r = httpx.post(f"{BASE}/reset", json={})
check("reset with {} body returns 200", r.status_code == 200, r.text[:200])
env_id = r.json().get("env_id", "")
obs    = r.json().get("observation", {})
check("reset returns env_id",           bool(env_id))
check("obs has alerts field",           "alerts"     in obs)
check("obs has recent_logs field",      "recent_logs" in obs)
check("obs has metrics field",          "metrics"    in obs)
check("obs has last_action_result",     "last_action_result" in obs)

# ── Test 2: /step with triage_alert ──────────────────────────────────────────
action = {
    "action_type": "triage_alert",
    "parameters":  {"alert_id": "a1", "severity": "P2", "team": "backend"},
}
r2 = httpx.post(f"{BASE}/step", json={"env_id": env_id, "action": action})
check("/step returns 200", r2.status_code == 200, r2.text[:200])
d2  = r2.json()
rew = d2.get("reward", {}).get("value", -999)
check("reward in [0.0, 1.0]", 0.0 <= rew <= 1.0, f"got {rew}")
check("step has done field",  "done" in d2)
check("step has observation", "observation" in d2)

# ── Test 3: /state includes step counter ─────────────────────────────────────
r3 = httpx.get(f"{BASE}/state/{env_id}")
check("/state returns 200", r3.status_code == 200, r3.text[:200])
st = r3.json()
check("state has step",     "step" in st,     f"keys: {list(st)}")
check("state has task",     "task" in st)
check("state has max_steps","max_steps" in st)

# ── Test 4: /tasks lists all 3 tasks ─────────────────────────────────────────
r4   = httpx.get(f"{BASE}/tasks")
check("/tasks returns 200", r4.status_code == 200)
ids  = [t["id"] for t in r4.json().get("tasks", [])]
check("3 tasks listed",         len(ids) == 3, f"got {ids}")
check("alert-triage present",        "alert-triage" in ids)
check("root-cause-analysis present", "root-cause-analysis" in ids)
check("full-incident-response present","full-incident-response" in ids)

# ── Test 5: RCA task hypothesis ───────────────────────────────────────────────
r5   = httpx.post(f"{BASE}/reset", json={"task": "root-cause-analysis", "seed": 42})
eid2 = r5.json()["env_id"]
hyp_action = {
    "action_type": "hypothesize",
    "parameters":  {"hypothesis": "payment-api memory leak caused by v2.4.1 depleted connection pool"},
}
r6  = httpx.post(f"{BASE}/step", json={"env_id": eid2, "action": hyp_action})
d6  = r6.json()
rew2 = d6.get("reward", {}).get("value", -999)
check("RCA reward in [0, 1]",   0.0 <= rew2 <= 1.0, f"got {rew2}")
check("RCA done after hypothesis", d6.get("done") is True)

# ── Test 6: Full-incident-response task ───────────────────────────────────────
r7   = httpx.post(f"{BASE}/reset", json={"task": "full-incident-response", "seed": 42})
check("full-response reset 200", r7.status_code == 200)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
else:
    print("All checks passed!")
