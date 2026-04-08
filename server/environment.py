"""
IncidentOps — Environment Core
================================
Manages episode state, action execution, and step-level reward shaping.
"""

from server.models import (
    IncidentAction, IncidentObservation, IncidentReward,
    ActionType, MetricSeries, DeployEvent,
)
from server.scenario_engine import ScenarioEngine
from typing import Dict, Any, Tuple
import copy


class IncidentOpsEnv:
    """
    Core IncidentOps environment.

    Lifecycle:
        env = IncidentOpsEnv(task="alert-triage", seed=42)
        obs = env.reset()
        obs, reward, done, info = env.step(action)
        state = env.state()
    """

    TASK_TEMPLATES = {
        "alert-triage":           "mixed_alerts",
        "root-cause-analysis":    "bad_deploy_memory_leak",
        "full-incident-response": "db_pool_exhaustion",
    }

    TASK_MAX_STEPS = {
        "alert-triage":           5,
        "root-cause-analysis":    12,
        "full-incident-response": 20,
    }

    def __init__(self, task: str, seed: int = 42):
        if task not in self.TASK_TEMPLATES:
            raise ValueError(
                f"Unknown task '{task}'. Valid: {list(self.TASK_TEMPLATES)}"
            )
        self.task      = task
        self.seed      = seed
        self.max_steps = self.TASK_MAX_STEPS[task]

        self.scenario_engine = ScenarioEngine()
        template_id          = self.TASK_TEMPLATES[task]
        self.scenario        = self.scenario_engine.generate(template_id, seed)

        # Episode state
        self.step_count      = 0
        self.agent_triages:  list  = []
        self.agent_hypothesis: str | None = None
        self.agent_fix:      str | None   = None
        self.state_data      = self._build_initial_state()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> IncidentObservation:
        """Reset the environment to a clean initial state (same task + seed)."""
        self.__init__(self.task, self.seed)
        return IncidentObservation(**self.state_data, step=0)

    def step(
        self, action: IncidentAction
    ) -> Tuple[IncidentObservation, IncidentReward, bool, Dict]:
        """
        Execute one action and return (observation, reward, done, info).

        - Reward is always clamped to [0.0, 1.0].
        - `done` becomes True when max_steps reached OR task-specific
          completion criteria are met (see _check_done).
        """
        self.step_count += 1

        result_msg  = self._execute_action(action)
        action_desc = f"{action.action_type.value}: {json_safe(action.parameters)}"
        self.state_data["last_action_result"] = result_msg
        self.state_data["incident_timeline"].append(
            f"Step {self.step_count}: {action_desc} → {result_msg}"
        )

        reward_value, breakdown = self._compute_reward(action)
        done = self._check_done()
        self.state_data["done"] = done

        obs    = IncidentObservation(**self.state_data, step=self.step_count)
        reward = IncidentReward(value=reward_value, breakdown=breakdown, done=done)
        return obs, reward, done, {}

    def state(self) -> Dict[str, Any]:
        """Return the full current state (debugging / /state endpoint)."""
        return {
            **self.state_data,
            "step":      self.step_count,
            "task":      self.task,
            "max_steps": self.max_steps,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_initial_state(self) -> Dict[str, Any]:
        """Strip grader-internal fields from alerts before exposing to agent."""
        obs_alerts = []
        for a in self.scenario["alerts"]:
            adict = copy.deepcopy(a)
            adict.pop("true_severity", None)
            adict.pop("true_team", None)
            obs_alerts.append(adict)

        metrics = self.scenario["metrics"]
        return {
            "alerts":             obs_alerts,
            "recent_logs":        self.scenario["logs"][:50],
            "metrics":            {k: v.model_dump() for k, v in metrics.items()},
            "deploy_history":     self.scenario["deploy_history"],
            "incident_timeline":  [],
            "resolution_status":  "open",
            "runbook_draft":      "",
            "comms_draft":        "",
            "last_action_result": "Incident started. Review alerts and begin investigation.",
            "done":               False,
        }

    def _check_done(self) -> bool:
        """Return True when episode should terminate."""
        # Hard cap always fires
        if self.step_count >= self.max_steps:
            return True

        # Task-specific early-done criteria
        if self.task == "alert-triage":
            # Done when agent has triaged every real alert OR step cap
            real_alert_count = sum(
                1 for a in self.scenario["alerts"] if "true_severity" in a
            )
            if len(self.agent_triages) >= real_alert_count:
                return True

        elif self.task == "root-cause-analysis":
            # Done as soon as a hypothesis is submitted
            if self.agent_hypothesis:
                return True

        elif self.task == "full-incident-response":
            # Done when incident is explicitly closed
            if self.state_data["resolution_status"] == "closed":
                return True

        return False

    def _execute_action(self, action: IncidentAction) -> str:
        at = action.action_type
        p  = action.parameters

        if at == ActionType.TRIAGE_ALERT:
            triage = {
                "alert_id": p.get("alert_id"),
                "severity":  str(p.get("severity", "")),
                "team":      str(p.get("team", "")),
            }
            # Avoid double-triaging the same alert
            already = [t for t in self.agent_triages if t["alert_id"] == triage["alert_id"]]
            if already:
                return f"Alert {triage['alert_id']} already triaged."
            self.agent_triages.append(triage)
            return (
                f"Alert {triage['alert_id']} classified as {triage['severity']} "
                f"→ routed to {triage['team']} team."
            )

        elif at == ActionType.QUERY_LOGS:
            srv = p.get("service", "")
            lvl = p.get("level", "")
            filtered = [
                log for log in self.scenario["logs"]
                if (not srv or log["service"] == srv)
                and (not lvl or log["level"] == lvl)
            ]
            self.state_data["recent_logs"] = filtered[-50:]
            return (
                f"Found {len(filtered)} log lines for service='{srv}' level='{lvl}'. "
                f"Showing last {min(50, len(filtered))}."
            )

        elif at == ActionType.QUERY_METRICS:
            srv = p.get("service", "")
            keys = [k for k in self.scenario["metrics"] if k.startswith(srv)]
            return f"Metrics for '{srv}': {', '.join(keys) or 'none found'}."

        elif at == ActionType.PAGE_TEAM:
            return f"Paged {p.get('team', 'unknown')} team. They have been notified."

        elif at == ActionType.HYPOTHESIZE:
            self.agent_hypothesis = p.get("hypothesis", "")
            return f"Root cause hypothesis recorded: «{self.agent_hypothesis[:120]}»"

        elif at == ActionType.APPLY_FIX:
            fix_id = p.get("fix_id", "")
            self.agent_fix = fix_id
            if fix_id in self.scenario.get("valid_fixes", []):
                self.state_data["resolution_status"] = "mitigated"
                return (
                    f"✅ Fix '{fix_id}' applied successfully. "
                    "Service metrics are recovering. Status → mitigated."
                )
            return (
                f"⚠️  Fix '{fix_id}' applied but no improvement observed. "
                "Check your root cause hypothesis."
            )

        elif at == ActionType.WRITE_RUNBOOK:
            text = p.get("text", "").strip()
            self.state_data["runbook_draft"] += ("\n\n" if self.state_data["runbook_draft"] else "") + text
            word_count = len(self.state_data["runbook_draft"].split())
            return f"Runbook updated ({word_count} words total)."

        elif at == ActionType.WRITE_COMMS:
            text = p.get("text", "").strip()
            self.state_data["comms_draft"] += ("\n\n" if self.state_data["comms_draft"] else "") + text
            return "Customer comms draft updated."

        elif at == ActionType.CLOSE_INCIDENT:
            if self.state_data["resolution_status"] != "mitigated":
                return "Cannot close incident — must apply a valid fix first (status must be 'mitigated')."
            self.state_data["resolution_status"] = "closed"
            return "Incident marked as closed."

        return "Action executed."

    def _compute_reward(self, action: IncidentAction) -> Tuple[float, Dict[str, float]]:
        """
        Step-level reward shaping.

        All values are contribution percentages of the final [0, 1] scale.
        The true final score is always computed by the grader at episode end.
        """
        reward    = 0.0
        breakdown = {}
        at = action.action_type
        p  = action.parameters

        if at == ActionType.TRIAGE_ALERT:
            alert_id = p.get("alert_id")
            alert    = next(
                (a for a in self.scenario["alerts"] if a.get("alert_id") == alert_id),
                None,
            )
            if alert and "true_severity" in alert:
                ag_sev = str(p.get("severity", ""))
                gt_sev = alert["true_severity"]
                ag_tm  = str(p.get("team", ""))
                gt_tm  = alert["true_team"]

                sev_ok = ag_sev == gt_sev
                # Partial credit for one severity step off
                sev_ranks = ["P0", "P1", "P2"]
                near_sev  = (
                    not sev_ok
                    and ag_sev in sev_ranks
                    and gt_sev in sev_ranks
                    and abs(sev_ranks.index(ag_sev) - sev_ranks.index(gt_sev)) == 1
                )

                breakdown["triage_severity"] = 0.04 if sev_ok else (0.02 if near_sev else 0.0)
                breakdown["triage_team"]     = 0.04 if (ag_tm == gt_tm) else 0.0
                reward += breakdown["triage_severity"] + breakdown["triage_team"]

        elif at == ActionType.QUERY_LOGS:
            relevant = self.scenario.get("relevant_services", [])
            srv      = p.get("service", "")
            if srv in relevant:
                breakdown["relevant_investigation"] = 0.03
                reward += 0.03
            # Penalise exact duplicate queries
            dup_count = sum(
                1 for line in self.state_data["incident_timeline"]
                if "query_logs" in line.lower() and srv in line
            )
            if dup_count > 1:
                breakdown["duplicate_query_penalty"] = -0.03
                reward -= 0.03

        elif at == ActionType.APPLY_FIX:
            if p.get("fix_id") in self.scenario.get("valid_fixes", []):
                breakdown["valid_fix"] = 0.20
                reward += 0.20
            else:
                breakdown["invalid_fix"] = -0.05
                reward -= 0.05

        elif at == ActionType.CLOSE_INCIDENT:
            if self.state_data["resolution_status"] == "closed":
                breakdown["clean_close"] = 0.05
                reward += 0.05

        # Always clamp to [0.0, 1.0] — never return negative per spec
        final = round(max(0.0, min(reward, 1.0)), 4)
        return final, breakdown


# ── Utility ───────────────────────────────────────────────────────────────────

def json_safe(obj: Any) -> str:
    """Convert parameters dict to a short string safely."""
    try:
        import json
        return json.dumps(obj, default=str)[:120]
    except Exception:
        return str(obj)[:120]


# Allow `from server.environment import *` imports to work cleanly
__all__ = ["IncidentOpsEnv"]
