"""
Data models for the Incidentops Environment.

Re-exports the typed Pydantic models used by the OpenEnv client interface.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Dict, Any, List, Literal, Optional


class IncidentopsAction(Action):
    """Action for the IncidentOps environment — an SRE action to take."""

    action_type: str = Field(..., description="Type of SRE action (e.g. triage_alert, query_logs, hypothesize, apply_fix)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters specific to the action_type")


class IncidentopsObservation(Observation):
    """Observation from the IncidentOps environment — current incident state."""

    step: int = Field(default=0, description="Current step number")
    alerts: List[Dict[str, Any]] = Field(default_factory=list, description="Active production alerts")
    recent_logs: List[Dict[str, Any]] = Field(default_factory=list, description="Recent log entries")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Time-series metrics")
    deploy_history: List[Dict[str, Any]] = Field(default_factory=list, description="Recent deployment events")
    incident_timeline: List[str] = Field(default_factory=list, description="Agent's action history")
    resolution_status: str = Field(default="open", description="Current incident status")
    runbook_draft: str = Field(default="", description="Accumulated runbook text")
    comms_draft: str = Field(default="", description="Accumulated customer comms text")
    last_action_result: str = Field(default="", description="Feedback from last action")
    done: bool = Field(default=False, description="Whether the episode is finished")
