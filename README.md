---
title: IncidentOps
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# IncidentOps — OpenEnv Environment

> **Production Incident Response & Root Cause Analysis Simulator**

IncidentOps is a real-world OpenEnv environment where an AI agent steps into the shoes of a Site Reliability Engineer (SRE) responding to a live production incident. The agent receives a stream of synthetic alerts, log lines, metric time-series, and deployment history — and must triage, investigate, identify the root cause, apply a fix, write a runbook, and communicate with customers.

Every tech company runs 24/7 on-call operations. Junior SREs can take **1–2 years** to become proficient at Root Cause Analysis across correlated, cascading failures. IncidentOps provides a safe, reproducible environment to train and evaluate AI agents on this high-stakes, genuinely hard task.

---

## Tasks

| Task ID | Difficulty | Max Steps | Description |
|---|---|---|---|
| `alert-triage` | 🟢 Easy | 5 | Classify 6 real alerts (mixed with noise) by severity (P0/P1/P2) and route to the correct on-call team |
| `root-cause-analysis` | 🟡 Medium | 12 | Query logs and metrics to identify the root service, failure mechanism, and triggering event |
| `full-incident-response` | 🔴 Hard | 20 | Complete SRE workflow: triage → investigate → hypothesize → apply fix → write runbook → customer comms → close |

### Difficulty Progression

**Easy (alert-triage):** The agent sees 6 real alerts mixed with 4–7 noise alerts. Real alerts span all severity levels (P0, P1, P2) and all 5 teams (infra, backend, data, frontend, security). A frontier LLM typically scores ~0.55–0.75.

**Medium (root-cause-analysis):** The agent must query logs and metrics across 3 microservices to find a cascading failure caused by a bad deploy or memory leak. It must submit a free-text hypothesis mentioning the root service, mechanism, and trigger. Frontier models with investigation typically score ~0.45–0.65.

**Hard (full-incident-response):** The agent must triage 4 real alerts across a cascading DB pool exhaustion, complete all 5 sub-tasks (triage, RCA, fix, runbook, comms), and close the incident. Missing any step strongly degrades the score. Partial credit is given for each completed sub-task. Frontier models typically score ~0.25–0.45.

---

## Action Space

The agent can take any of these actions each step:

| Action | Parameters | Description |
|---|---|---|
| `triage_alert` | `alert_id`, `severity` (P0/P1/P2), `team` | Classify one alert and route it |
| `query_logs` | `service`, `level` (optional) | Filter log lines by service and/or level |
| `query_metrics` | `service` | View metric time-series for a service |
| `page_team` | `team` | Escalate to an on-call team |
| `hypothesize` | `hypothesis` | Submit root cause explanation (ends RCA task) |
| `apply_fix` | `fix_id` | Apply a remediation action |
| `write_runbook` | `text` | Append a section to the runbook draft |
| `write_comms` | `text` | Append a customer-facing status update |
| `close_incident` | _(none)_ | Mark the incident as resolved (ends full-response task) |

**Available teams:** `infra`, `backend`, `data`, `frontend`, `security`

---

## Observation Space

Each step the agent receives an `IncidentObservation` with:

| Field | Type | Description |
|---|---|---|
| `alerts` | `List[Alert]` | All active alerts (severity/team hidden, agent must infer) |
| `recent_logs` | `List[LogEntry]` | Last 50 log lines (pageable via `query_logs`) |
| `metrics` | `Dict[str, MetricSeries]` | Time-series for each service×metric (60-minute window) |
| `deploy_history` | `List[DeployEvent]` | Last 10 CI/CD deploys — may contain the bad deploy |
| `incident_timeline` | `List[str]` | Agent's own action history (breadcrumb) |
| `resolution_status` | `str` | `open` → `mitigated` → `closed` |
| `runbook_draft` | `str` | Accumulated runbook text |
| `comms_draft` | `str` | Accumulated customer comms text |
| `last_action_result` | `str` | Natural-language feedback on the last action taken |
| `done` | `bool` | Whether the episode has ended |

---

## Reward Function

Rewards are always in `[0.0, 1.0]`. The environment uses **shaped rewards** during the episode plus a **final grader score** at episode end.

### Step-level shaping

| Event | Reward |
|---|---|
| Correct severity triage | +0.04 |
| One-off severity triage (e.g. P1 instead of P0) | +0.02 |
| Correct team routing | +0.04 |
| Querying a relevant service's logs | +0.03 |
| Applying a valid fix | +0.20 |
| Applying an invalid fix | +0.00 (no negative penalty, clamped) |
| Clean close after mitigated | +0.05 |
| Duplicate log query | −0.03 (but floors at 0.0) |

### Final grader (replaces step reward at `done=True`)

| Task | Grader |
|---|---|
| `alert-triage` | `0.5 × severity_accuracy + 0.5 × team_accuracy` |
| `root-cause-analysis` | Keyword matching: required terms (80%) + supporting terms (20% bonus) |
| `full-incident-response` | `0.20 × triage + 0.35 × rca + 0.20 × fix + 0.15 × runbook + 0.10 × comms` |

---

## Setup & Usage

### Prerequisites

- Python 3.11+
- Docker (for containerised deployment)
- A Hugging Face account or OpenAI-compatible API key

### Local Development

```bash
git clone <your-repo>
cd incidentops
pip install -r requirements.txt

# Start the environment server
uvicorn server.main:app --host 0.0.0.0 --port 7860 --reload
```

The API docs are available at: **(https://huggingface.co/spaces/SERG4NT/incidentops)**

### Docker

```bash
# Build
docker build -t incidentops .

# Run
docker run -p 7860:7860 incidentops

# Test the /reset endpoint
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "alert-triage", "seed": 42}'
```

### Running the Baseline Inference Script

Set these environment variables, then run `inference.py`. The script loops over **all 3 tasks** automatically.

**Linux / macOS:**
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"
python inference.py
```

**Windows (PowerShell):**
```powershell
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME   = "Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN     = "hf_your_token_here"
python inference.py
```

The script emits structured logs in the required format:
```
[START] task=alert-triage env=incidentops model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"action_type":"triage_alert",...} reward=0.08 done=false error=null
[STEP] step=2 action={"action_type":"triage_alert",...} reward=0.08 done=false error=null
[END] success=true steps=2 score=0.622 rewards=0.08,0.08
[START] task=root-cause-analysis env=incidentops model=Qwen/Qwen2.5-72B-Instruct
...
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness probe |
| `/tasks` | GET | List all tasks with metadata |
| `/reset` | POST | Start a new episode → returns `env_id` + initial observation |
| `/step` | POST | Execute one action → returns observation, reward, done |
| `/state/{env_id}` | GET | Get full current state (debugging) |
| `/docs` | GET | Interactive Swagger UI |

### Quick example (curl)

```bash
# 1. Reset
ENV_ID=$(curl -s -X POST http://localhost:7860/reset \
  -H 'Content-Type: application/json' \
  -d '{}' | python -c "import sys,json; print(json.load(sys.stdin)['env_id'])")

# 2. Take a step
curl -X POST http://localhost:7860/step \
  -H 'Content-Type: application/json' \
  -d "{\"env_id\":\"$ENV_ID\",\"action\":{\"action_type\":\"triage_alert\",\"parameters\":{\"alert_id\":\"a1\",\"severity\":\"P0\",\"team\":\"infra\"}}}"
```

---

## Baseline Scores

Reproduced with `Qwen/Qwen2.5-72B-Instruct` at `seed=42`, `temperature=0.2`:

| Task | Score | Steps Used |
|---|---|---|
| `alert-triage` | ~0.62 | 2 |
| `root-cause-analysis` | ~0.50 | 7 |
| `full-incident-response` | ~0.38 | 18 |
| **Average** | **~0.50** | |

---

## Project Structure

```
incidentops/
├── inference.py          # Baseline inference script (run this)
├── openenv.yaml          # OpenEnv spec metadata
├── Dockerfile            # Container definition
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── scenarios/            # Scenario JSON templates
│   ├── mixed_alerts.json          # alert-triage scenario
│   ├── bad_deploy_memory_leak.json # root-cause-analysis scenario
│   └── db_pool_exhaustion.json    # full-incident-response scenario
└── server/               # Environment server
    ├── main.py           # FastAPI app with /reset, /step, /state
    ├── models.py         # Typed Pydantic models (Observation, Action, Reward)
    ├── environment.py    # Core environment logic
    ├── scenario_engine.py # Procedural scenario generation
    └── graders/          # Deterministic task graders
        ├── triage_grader.py
        ├── rca_grader.py
        └── full_response_grader.py
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | _(required)_ | Hugging Face / API key |
| `INCIDENTOPS_URL` | `http://localhost:7860` | Environment server URL |

---

## License

MIT
