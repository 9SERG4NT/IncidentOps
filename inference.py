"""
IncidentOps — Baseline Inference Script
========================================
Runs the IncidentOps OpenEnv environment against a language model across
all three tasks (alert-triage → root-cause-analysis → full-incident-response)
and emits structured logs in the mandatory OpenEnv format.

Environment variables (set before running):
    API_BASE_URL   LLM endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model to use  (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       Hugging Face / API key

Quick start:
    # Linux/macOS
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
    export HF_TOKEN="hf_..."
    python inference.py

    # Windows PowerShell
    $env:API_BASE_URL="https://router.huggingface.co/v1"
    $env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
    $env:HF_TOKEN="hf_..."
    python inference.py

Output format (one line each, strictly):
    [START] task=<name> env=incidentops model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import json
import textwrap
import httpx
from typing import Any, Dict, List, Optional
from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME:   str = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY:      str = os.getenv("HF_TOKEN", os.getenv("API_KEY", "dummy"))

ENV_BASE_URL:  str = os.getenv("INCIDENTOPS_URL", "http://localhost:7860")
IMAGE_NAME:    str = os.getenv("IMAGE_NAME", "incidentops")
LOCAL_IMAGE_NAME: str = os.getenv("LOCAL_IMAGE_NAME", IMAGE_NAME)
BENCHMARK:     str = "incidentops"
SEED:          int = 42

# Per-task step budgets (must stay within openenv.yaml max_steps)
TASK_CONFIG = {
    "alert-triage":          {"max_steps": 5,  "success_threshold": 0.40},
    "root-cause-analysis":   {"max_steps": 10, "success_threshold": 0.50},
    "full-incident-response":{"max_steps": 18, "success_threshold": 0.40},
}

TEMPERATURE = 0.2   # Low temperature → more deterministic, reproducible scores
MAX_TOKENS  = 300

# ── OpenEnv HTTP Client ───────────────────────────────────────────────────────
class IncidentOpsClient:
    """Thin HTTP wrapper around the IncidentOps REST environment."""

    def __init__(self, base_url: str = ENV_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(timeout=30.0)
        self.env_id: Optional[str] = None

    def reset(self, task: str, seed: int = SEED) -> Dict[str, Any]:
        resp = self.http.post(
            f"{self.base_url}/reset",
            json={"task": task, "seed": seed},
        )
        resp.raise_for_status()
        data = resp.json()
        self.env_id = data["env_id"]
        return data["observation"]

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        resp = self.http.post(
            f"{self.base_url}/step",
            json={"env_id": self.env_id, "action": action},
        )
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        resp = self.http.get(f"{self.base_url}/state/{self.env_id}")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self.http.close()

# ── Mandatory Log Helpers ────────────────────────────────────────────────────
def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── System Prompts (one per task) ────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "alert-triage": textwrap.dedent("""
        You are a senior SRE (Site Reliability Engineer) triaging production alerts.
        Your job: classify each alert with the correct severity and route it to the right team.

        Available severities: P0 (critical, page now), P1 (high, 15-min SLA), P2 (medium, next biz hour)
        Available teams: infra, backend, data, frontend, security

        You MUST respond with a JSON action object — no other text:
        {
          "action_type": "triage_alert",
          "parameters": {
            "alert_id": "<id from observation>",
            "severity": "<P0|P1|P2>",
            "team": "<infra|backend|data|frontend|security>"
          }
        }

        Triage logic:
        - Database/infrastructure alerts → infra team
        - High error rates, API failures → backend team
        - UI/load time issues → frontend team
        - Data pipeline issues → data team
        - P0: service down or >5% error rate or DB exhausted
        - P1: degraded performance, elevated errors
        - P2: minor spikes, warnings
    """).strip(),

    "root-cause-analysis": textwrap.dedent("""
        You are a senior SRE performing root cause analysis on a production incident.
        Investigate logs and metrics, then state your root cause hypothesis.

        Available action types:
        1. Query logs:    {"action_type": "query_logs",   "parameters": {"service": "<name>", "level": "ERROR"}}
        2. Query metrics: {"action_type": "query_metrics","parameters": {"service": "<name>"}}
        3. Hypothesize:   {"action_type": "hypothesize",  "parameters": {"hypothesis": "<your root cause explanation>"}}

        Respond with exactly one JSON action object — no other text.

        Investigation strategy:
        - Start by querying ERROR logs for the most affected services
        - Look for correlation between deploy timestamps and anomaly start
        - Check metrics for memory/cpu spikes that precede cascading failures
        - Your final hypothesis MUST mention: the specific service, the failure mechanism, and the triggering event (e.g., bad deploy version)
    """).strip(),

    "full-incident-response": textwrap.dedent("""
        You are the incident commander for a production P0 incident. You must fully resolve it.

        Available actions (respond with exactly one JSON object per turn):
        1. Triage:    {"action_type": "triage_alert",   "parameters": {"alert_id":"<id>","severity":"<P0|P1|P2>","team":"<team>"}}
        2. Query logs:{"action_type": "query_logs",     "parameters": {"service": "<name>", "level": "ERROR"}}
        3. Hypothesize:{"action_type":"hypothesize",    "parameters": {"hypothesis": "<root cause>"}}
        4. Apply fix: {"action_type": "apply_fix",      "parameters": {"fix_id": "<one of the valid fix IDs from scenario>"}}
        5. Runbook:   {"action_type": "write_runbook",  "parameters": {"text": "<section text>"}}
        6. Comms:     {"action_type": "write_comms",    "parameters": {"text": "<customer update>"}}
        7. Close:     {"action_type": "close_incident", "parameters": {}}

        Full workflow you MUST complete in order:
        Step 1-2: Triage the P0 alerts
        Step 3-4: Query ERROR logs for affected services to confirm root cause
        Step 5:   Hypothesize with specific root cause (service + mechanism + trigger)
        Step 6:   Apply the correct fix (check deploy history for which service to fix)
        Step 7:   Write a runbook with sections: symptoms, root cause, fix, prevention
        Step 8:   Write customer comms mentioning mitigation/monitoring/resolution
        Step 9:   Close the incident

        Do NOT repeat the same log query twice. Each action must make forward progress.
    """).strip(),
}

# ── Action Tool Schema ────────────────────────────────────────────────────────
ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "take_action",
        "description": "Take an action in the IncidentOps environment",
        "parameters": {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": [
                        "triage_alert", "query_logs", "query_metrics",
                        "page_team", "hypothesize", "apply_fix",
                        "write_runbook", "write_comms", "close_incident",
                    ],
                    "description": "The type of SRE action to take",
                },
                "parameters": {
                    "type": "object",
                    "description": "Parameters specific to the action_type",
                },
            },
            "required": ["action_type", "parameters"],
        },
    },
}

# Task-specific fallback actions when LLM fails
FALLBACK_ACTIONS = {
    "alert-triage":           {"action_type": "triage_alert",  "parameters": {"alert_id": "a1", "severity": "P1", "team": "backend"}},
    "root-cause-analysis":    {"action_type": "hypothesize",   "parameters": {"hypothesis": "Unknown error in the system."}},
    "full-incident-response": {"action_type": "close_incident","parameters": {}},
}

# ── LLM Call ─────────────────────────────────────────────────────────────────
def get_action(
    llm: OpenAI,
    task: str,
    observation: Dict[str, Any],
    history: List[Dict],
) -> Dict[str, Any]:
    """Ask the LLM for the next action. Returns a valid action dict."""
    obs_str = json.dumps(observation, default=str, ensure_ascii=False)[:3000]  # cap length

    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[task]},
        *history[-6:],  # last 6 turns of context
        {"role": "user",   "content": f"Current observation:\n{obs_str}\n\nWhat is your next action?"},
    ]

    try:
        resp = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=[ACTION_TOOL],
            tool_choice={"type": "function", "function": {"name": "take_action"}},
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            return json.loads(msg.tool_calls[0].function.arguments)
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)

    return FALLBACK_ACTIONS.get(task, {"action_type": "close_incident", "parameters": {}})


# ── Single Task Runner ────────────────────────────────────────────────────────
def run_task(task: str, llm: OpenAI) -> float:
    """Run one full episode for `task`. Returns final score in [0.0, 1.0]."""
    cfg        = TASK_CONFIG[task]
    max_steps  = cfg["max_steps"]
    threshold  = cfg["success_threshold"]

    env     = IncidentOpsClient()
    rewards: List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False
    history: List[Dict] = []

    log_start(task=task, model=MODEL_NAME)

    try:
        obs = env.reset(task=task, seed=SEED)

        for step in range(1, max_steps + 1):
            # Check if already done from previous step
            if obs.get("done", False):
                break

            action = get_action(llm, task, obs, history)
            action_str = json.dumps(action, separators=(",", ":"))

            error_msg: Optional[str] = None
            try:
                result   = env.step(action)
                obs      = result["observation"]
                reward_v = float(result["reward"]["value"])
                done     = bool(result["done"])
            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP_{e.response.status_code}"
                reward_v  = 0.0
                done      = False
            except Exception as e:
                error_msg = str(e)[:80].replace("\n", " ")
                reward_v  = 0.0
                done      = False

            # Clamp reward to [0, 1] per spec
            reward_v = max(0.0, min(reward_v, 1.0))

            rewards.append(reward_v)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward_v, done=done, error=error_msg)

            # Update conversation history for LLM context
            history.append({"role": "assistant", "content": f"Action taken: {action_str}"})
            history.append({"role": "user",      "content": f"Result: reward={reward_v:.2f}, done={done}, last_action_result={obs.get('last_action_result', '')}"})

            if done:
                break

        # Score = final grader reward (server replaces last reward with grader score)
        score   = round(rewards[-1], 3) if rewards else 0.0
        score   = max(0.0, min(score, 1.0))
        success = score >= threshold

    except Exception as exc:
        print(f"[DEBUG] Episode crashed: {exc}", flush=True)
    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores: Dict[str, float] = {}
    for task in TASK_CONFIG:
        score = run_task(task, llm)
        all_scores[task] = score
        print(f"[DEBUG] {task} => score={score:.3f}", flush=True)
        print("", flush=True)  # blank line between tasks for readability

    print("[DEBUG] === Final Scores ===", flush=True)
    for task, s in all_scores.items():
        print(f"[DEBUG]   {task}: {s:.3f}", flush=True)
    print(f"[DEBUG] Average: {sum(all_scores.values())/len(all_scores):.3f}", flush=True)


if __name__ == "__main__":
    main()
