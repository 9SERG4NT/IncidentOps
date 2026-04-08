"""Comprehensive test of all IncidentOps endpoints and scenarios."""
import httpx
import json
import sys
import traceback

BASE = "http://localhost:7860"
client = httpx.Client(timeout=30)
errors = []

def test(name, func):
    print(f"TEST: {name}")
    try:
        func()
        print(f"  PASSED")
    except Exception as e:
        traceback.print_exc()
        errors.append((name, str(e)))
        print(f"  FAILED: {e}")

def test_health():
    r = client.get(f"{BASE}/health")
    assert r.status_code == 200

def test_reset_all_tasks():
    for task in ["alert-triage", "root-cause-analysis", "full-incident-response"]:
        r = client.post(f"{BASE}/reset", json={"task": task, "seed": 42})
        assert r.status_code == 200, f"Reset failed for {task}: {r.text}"
        data = r.json()
        obs = data["observation"]
        n_alerts = len(obs.get("alerts", []))
        n_logs = len(obs.get("recent_logs", []))
        n_metrics = len(obs.get("metrics", {}))
        n_deploys = len(obs.get("deploy_history", []))
        print(f"  {task}: {n_alerts} alerts, {n_logs} logs, {n_metrics} metrics, {n_deploys} deploys")

def test_alert_triage():
    r = client.post(f"{BASE}/reset", json={"task": "alert-triage", "seed": 42})
    data = r.json()
    env_id = data["env_id"]
    obs = data["observation"]
    alerts = obs["alerts"]
    
    # Check no ground truth leakage
    for a in alerts:
        assert "true_severity" not in a, "true_severity leaked for " + a["alert_id"]
        assert "true_team" not in a, "true_team leaked for " + a["alert_id"]
    print(f"  No ground truth leakage in {len(alerts)} alerts")
    
    # Triage both real alerts correctly
    a1 = {"action_type": "triage_alert", "parameters": {"alert_id": "a1", "severity": "P2", "team": "backend"}}
    r2 = client.post(f"{BASE}/step", json={"env_id": env_id, "action": a1})
    assert r2.status_code == 200
    s1 = r2.json()
    print(f"  Step1 reward: {s1['reward']['value']}, done: {s1['done']}")
    
    a2 = {"action_type": "triage_alert", "parameters": {"alert_id": "a2", "severity": "P2", "team": "frontend"}}
    r3 = client.post(f"{BASE}/step", json={"env_id": env_id, "action": a2})
    assert r3.status_code == 200
    s2 = r3.json()
    print(f"  Step2 reward: {s2['reward']['value']}, done: {s2['done']}")
    assert s2["done"] == True, "Should be done after triaging all real alerts"
    print(f"  Final triage score: {s2['reward']['value']}")

def test_rca():
    r = client.post(f"{BASE}/reset", json={"task": "root-cause-analysis", "seed": 42})
    data = r.json()
    env_id = data["env_id"]
    
    action = {"action_type": "query_logs", "parameters": {"service": "payment-api"}}
    r2 = client.post(f"{BASE}/step", json={"env_id": env_id, "action": action})
    print(f"  Query logs reward: {r2.json()['reward']['value']}")
    
    action2 = {"action_type": "hypothesize", "parameters": {"hypothesis": "The payment-api experienced a memory leak after deploying v2.4.1, causing connection pool exhaustion and OOM."}}
    r3 = client.post(f"{BASE}/step", json={"env_id": env_id, "action": action2})
    s = r3.json()
    print(f"  Hypothesis reward: {s['reward']['value']}, done: {s['done']}")
    assert s["done"] == True

def test_full_response():
    r = client.post(f"{BASE}/reset", json={"task": "full-incident-response", "seed": 42})
    data = r.json()
    env_id = data["env_id"]
    
    steps = [
        {"action_type": "triage_alert", "parameters": {"alert_id": "a1", "severity": "P0", "team": "infra"}},
        {"action_type": "query_logs", "parameters": {"service": "db-cluster"}},
        {"action_type": "hypothesize", "parameters": {"hypothesis": "db-cluster connection pool exhausted due to payment-api traffic surge"}},
        {"action_type": "apply_fix", "parameters": {"fix_id": "restart_db"}},
        {"action_type": "write_runbook", "parameters": {"text": "Symptoms: DB connection pool exhausted. Root Cause: payment-api traffic. Fix: Restart db. Prevention: Add monitoring alerts."}},
        {"action_type": "write_comms", "parameters": {"text": "We have mitigated the database connection issue. Monitoring for recovery."}},
        {"action_type": "close_incident", "parameters": {}},
    ]
    for i, action in enumerate(steps):
        r2 = client.post(f"{BASE}/step", json={"env_id": env_id, "action": action})
        s = r2.json()
        print(f"  Step {i+1} ({action['action_type']}): reward={s['reward']['value']}, done={s['done']}")
    assert s["done"] == True

def test_determinism():
    r1 = client.post(f"{BASE}/reset", json={"task": "root-cause-analysis", "seed": 42})
    r2 = client.post(f"{BASE}/reset", json={"task": "root-cause-analysis", "seed": 42})
    obs1 = r1.json()["observation"]
    obs2 = r2.json()["observation"]
    assert obs1["alerts"] == obs2["alerts"]
    assert obs1["recent_logs"] == obs2["recent_logs"]
    assert obs1["deploy_history"] == obs2["deploy_history"]
    for k in obs1["metrics"]:
        assert obs1["metrics"][k]["values"] == obs2["metrics"][k]["values"], f"Metric {k} not deterministic"
    print("  Deterministic: identical observations for same seed")

def test_done_enforcement():
    r = client.post(f"{BASE}/reset", json={"task": "root-cause-analysis", "seed": 42})
    env_id = r.json()["env_id"]
    action = {"action_type": "hypothesize", "parameters": {"hypothesis": "Test"}}
    client.post(f"{BASE}/step", json={"env_id": env_id, "action": action})
    r3 = client.post(f"{BASE}/step", json={"env_id": env_id, "action": action})
    assert r3.status_code == 400
    print(f"  Post-done step correctly rejected: {r3.json()['detail']}")

def test_state_endpoint():
    r = client.post(f"{BASE}/reset", json={"task": "alert-triage", "seed": 42})
    env_id = r.json()["env_id"]
    r2 = client.get(f"{BASE}/state/{env_id}")
    assert r2.status_code == 200
    state = r2.json()
    print(f"  State keys: {list(state.keys())}")
    r3 = client.get(f"{BASE}/state/nonexistent")
    assert r3.status_code == 404

def test_invalid_action():
    r = client.post(f"{BASE}/reset", json={"task": "alert-triage", "seed": 42})
    env_id = r.json()["env_id"]
    action = {"action_type": "triage_alert", "parameters": {}}
    r2 = client.post(f"{BASE}/step", json={"env_id": env_id, "action": action})
    assert r2.status_code == 422, f"Expected 422, got {r2.status_code}"
    print("  Invalid action correctly rejected with 422")

def test_invalid_task():
    r = client.post(f"{BASE}/reset", json={"task": "nonexistent-task", "seed": 42})
    assert r.status_code in [400, 422, 500], f"Expected error, got {r.status_code}"
    print(f"  Invalid task correctly returned {r.status_code}")

tests = [
    ("Health Check", test_health),
    ("Reset All Tasks", test_reset_all_tasks),
    ("Alert Triage Playthrough", test_alert_triage),
    ("RCA Playthrough", test_rca),
    ("Full Incident Response", test_full_response),
    ("Determinism", test_determinism),
    ("Done Enforcement", test_done_enforcement),
    ("State Endpoint", test_state_endpoint),
    ("Invalid Action", test_invalid_action),
    ("Invalid Task", test_invalid_task),
]

for name, func in tests:
    test(name, func)

passed = len(tests) - len(errors)
print(f"\nRESULTS: {passed}/{len(tests)} passed")
if errors:
    print("FAILURES:")
    for name, err in errors:
        print(f"  FAILED: {name}: {err}")
sys.exit(0 if not errors else 1)
