from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from server.models import IncidentAction, IncidentObservation, IncidentReward
from server.environment import IncidentOpsEnv
import uuid

app = FastAPI(
    title="IncidentOps OpenEnv",
    description=(
        "Production incident response and root cause analysis environment. "
        "An AI agent takes the role of an SRE triaging alerts, "
        "investigating logs/metrics, identifying root causes, and drafting runbooks."
    ),
    version="1.0.0",
)

# Allow cross-origin requests (needed for HF Spaces + browser clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store of active environments (keyed by env_id)
_envs: dict[str, IncidentOpsEnv] = {}

VALID_TASKS = {"alert-triage", "root-cause-analysis", "full-incident-response"}


# ── Request / Response models ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "alert-triage"   # default → validator's {} POST works
    seed: int = 42


class StepRequest(BaseModel):
    env_id: str
    action: IncidentAction


class StepResponse(BaseModel):
    observation: IncidentObservation
    reward: IncidentReward
    done: bool
    info: dict = {}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness probe."""
    return {"status": "ok", "active_envs": len(_envs)}


@app.get("/tasks")
def list_tasks():
    """Return all available task IDs with metadata."""
    return {
        "tasks": [
            {
                "id": "alert-triage",
                "difficulty": "easy",
                "max_steps": 5,
                "description": "Classify mixed-severity alerts and route each to the correct on-call team.",
            },
            {
                "id": "root-cause-analysis",
                "difficulty": "medium",
                "max_steps": 12,
                "description": "Given logs and metric anomalies, identify the root cause from evidence.",
            },
            {
                "id": "full-incident-response",
                "difficulty": "hard",
                "max_steps": 20,
                "description": "Full SRE workflow: triage → investigate → hypothesize → fix → runbook → comms → close.",
            },
        ]
    }


@app.post("/reset")
def reset(req: ResetRequest) -> dict:
    """
    Start a new environment episode.

    - Accepts `{}` body (uses defaults: task=alert-triage, seed=42).
    - Returns `env_id` for use in subsequent /step and /state calls.
    """
    if req.task not in VALID_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{req.task}'. Valid: {sorted(VALID_TASKS)}",
        )
    try:
        env = IncidentOpsEnv(task=req.task, seed=req.seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    env_id = str(uuid.uuid4())
    _envs[env_id] = env
    obs = env.reset()
    return {"env_id": env_id, "observation": obs.model_dump()}


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    """
    Advance the environment by one step.

    - `env_id` must come from a previous /reset call.
    - `action` must be a valid `IncidentAction` (see /docs for schema).
    """
    env = _envs.get(req.env_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"env_id '{req.env_id}' not found. Call /reset first.")

    if env.state_data.get("done", False):
        raise HTTPException(status_code=400, detail="Episode is already done. Call /reset to start a new episode.")

    obs, reward, done, info = env.step(req.action)

    # Run final grader when episode ends to replace shaping reward with true score
    if done:
        if env.task == "alert-triage":
            from server.graders.triage_grader import grade_triage
            gt = [a for a in env.scenario.get("alerts", []) if "true_severity" in a]
            reward.value = grade_triage(env.agent_triages, gt)

        elif env.task == "root-cause-analysis":
            from server.graders.rca_grader import grade_rca
            reward.value = grade_rca(
                env.agent_hypothesis or "",
                env.scenario.get("id", ""),
                env.scenario.get("root_cause_keywords", {}),
            )

        elif env.task == "full-incident-response":
            from server.graders.full_response_grader import grade_full_response
            reward.value = grade_full_response(env)

    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state/{env_id}")
def state(env_id: str) -> dict:
    """
    Return the full current state of an environment.

    Useful for debugging or building a custom client. The state includes
    everything the agent has seen plus internal counters.
    """
    env = _envs.get(env_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"env_id '{env_id}' not found.")
    return env.state()
