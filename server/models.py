"""
IncidentOps — Typed Pydantic Models
=====================================
Defines the Observation, Action, and Reward models used by the environment.
All models are strict Pydantic v2 BaseModels with field documentation.
"""

from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Literal, Optional, List, Dict, Any
from enum import Enum


# ── Enums ─────────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    """Alert severity levels aligned with standard SRE on-call tiers."""
    P0 = "P0"   # Critical — page the on-call immediately
    P1 = "P1"   # High     — 15-minute SLA response
    P2 = "P2"   # Medium   — next business hour


class Team(str, Enum):
    """On-call teams available for alert routing."""
    INFRA    = "infra"
    BACKEND  = "backend"
    DATA     = "data"
    FRONTEND = "frontend"
    SECURITY = "security"


# ── Component Models ──────────────────────────────────────────────────────────

class Alert(BaseModel):
    """
    A single production alert.
    `true_severity` and `true_team` are hidden from the agent (grader-only).
    """
    alert_id:       str
    title:          str
    service:        str
    timestamp:      str
    raw_value:      float
    threshold:      float
    true_severity:  Optional[Severity] = Field(default=None, exclude=True)   # grader-only
    true_team:      Optional[Team]     = Field(default=None, exclude=True)   # grader-only


class LogEntry(BaseModel):
    """A single structured log line from a service."""
    ts:       str
    level:    Literal["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]
    service:  str
    message:  str
    trace_id: Optional[str] = None


class MetricSeries(BaseModel):
    """A time-series metric for a single service.metric combination."""
    name:       str
    unit:       str
    timestamps: List[str]
    values:     List[float]


class DeployEvent(BaseModel):
    """A single deployment event from the CI/CD pipeline."""
    deploy_id:    str
    service:      str
    sha:          str
    timestamp:    str
    deployer:     str
    diff_summary: str


# ── Observation ───────────────────────────────────────────────────────────────

class IncidentObservation(BaseModel):
    """
    Full observation returned to the agent each step.

    The agent sees:
    - All active alerts (without grader-internal fields)
    - Recent log lines (pageable via query_logs action)
    - Time-series metrics for each service
    - Deployment history (last 10 deploys)
    - The agent's own incident timeline (breadcrumb of past actions)
    - Current runbook and customer comms drafts
    - Feedback from the last action taken
    """
    step:               int
    alerts:             List[Alert]              # severity/team hidden from agent
    recent_logs:        List[LogEntry]           # last 50 log lines, pageable
    metrics:            Dict[str, MetricSeries]  # key = "service.metric_name"
    deploy_history:     List[DeployEvent]        # last 10 deploys across services
    incident_timeline:  List[str]                # agent's own action history as text
    resolution_status:  Literal["open", "mitigated", "closed"]
    runbook_draft:      str                      # accumulated runbook text
    comms_draft:        str                      # accumulated customer comms text
    last_action_result: str                      # natural-language feedback on last action
    done:               bool


# ── Action ────────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    """All available SRE actions the agent can take."""
    TRIAGE_ALERT  = "triage_alert"    # Classify one alert: severity + team
    QUERY_LOGS    = "query_logs"      # Filter logs by service / level / time
    QUERY_METRICS = "query_metrics"   # Pull a specific metric window
    PAGE_TEAM     = "page_team"       # Escalate to an on-call team
    HYPOTHESIZE   = "hypothesize"     # State a root cause theory
    APPLY_FIX     = "apply_fix"       # Choose a remediation action
    WRITE_RUNBOOK = "write_runbook"   # Append text to the runbook draft
    WRITE_COMMS   = "write_comms"     # Append text to the customer-facing comms
    CLOSE_INCIDENT= "close_incident"  # Mark the incident resolved


class IncidentAction(BaseModel):
    """
    A single agent action.

    Required `parameters` vary by `action_type`:

    | action_type      | required parameters                              |
    |------------------|--------------------------------------------------|
    | triage_alert     | alert_id, severity (P0/P1/P2), team             |
    | query_logs       | service, (optional) level                        |
    | query_metrics    | service                                          |
    | page_team        | team                                             |
    | hypothesize      | hypothesis (free-text root cause explanation)    |
    | apply_fix        | fix_id (must match a valid fix from the scenario)|
    | write_runbook    | text (markdown section to append)               |
    | write_comms      | text (customer-facing status update)             |
    | close_incident   | (no parameters required)                         |
    """
    action_type: ActionType
    parameters:  Dict[str, Any] = Field(default_factory=dict)

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: Dict[str, Any], info: ValidationInfo) -> Dict[str, Any]:
        at = info.data.get("action_type")
        if at == ActionType.TRIAGE_ALERT:
            missing = [k for k in ("alert_id", "severity", "team") if k not in v]
            if missing:
                raise ValueError(f"triage_alert requires: {missing}")
            try:
                v["severity"] = Severity(v["severity"])
                v["team"]     = Team(v["team"])
            except ValueError as e:
                raise ValueError(f"Invalid severity or team: {e}")
        elif at == ActionType.QUERY_LOGS:
            if "service" not in v:
                raise ValueError("query_logs requires: ['service']")
        elif at == ActionType.HYPOTHESIZE:
            if "hypothesis" not in v:
                raise ValueError("hypothesize requires: ['hypothesis']")
        elif at == ActionType.APPLY_FIX:
            if "fix_id" not in v:
                raise ValueError("apply_fix requires: ['fix_id']")
        return v


# ── Reward ────────────────────────────────────────────────────────────────────

class IncidentReward(BaseModel):
    """
    Reward signal returned after each step.

    `value` is always in [0.0, 1.0] per OpenEnv spec.
    `breakdown` shows the sub-component contributions for transparency.
    """
    value:     float            = Field(ge=0.0, le=1.0)   # clamped to [0, 1]
    breakdown: Dict[str, float] = Field(default_factory=dict)
    done:      bool
    info:      Dict[str, Any]   = Field(default_factory=dict)
