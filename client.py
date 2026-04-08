"""Incidentops Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import IncidentopsAction, IncidentopsObservation


class IncidentopsEnv(
    EnvClient[IncidentopsAction, IncidentopsObservation, State]
):
    """
    Client for the IncidentOps Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with IncidentopsEnv(base_url="http://localhost:7860") as client:
        ...     result = client.reset()
        ...     print(result.observation)
        ...
        ...     result = client.step(IncidentopsAction(
        ...         action_type="triage_alert",
        ...         parameters={"alert_id": "a1", "severity": "P0", "team": "infra"}
        ...     ))
        ...     print(result.observation.resolution_status)

    Example with Docker:
        >>> client = IncidentopsEnv.from_docker_image("incidentops:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(IncidentopsAction(
        ...         action_type="query_logs",
        ...         parameters={"service": "api-gateway", "level": "ERROR"}
        ...     ))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: IncidentopsAction) -> Dict:
        """
        Convert IncidentopsAction to JSON payload for step message.

        Args:
            action: IncidentopsAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type,
            "parameters": action.parameters,
        }

    def _parse_result(self, payload: Dict) -> StepResult[IncidentopsObservation]:
        """
        Parse server response into StepResult[IncidentopsObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with IncidentopsObservation
        """
        obs_data = payload.get("observation", {})
        observation = IncidentopsObservation(
            step=obs_data.get("step", 0),
            alerts=obs_data.get("alerts", []),
            recent_logs=obs_data.get("recent_logs", []),
            metrics=obs_data.get("metrics", {}),
            deploy_history=obs_data.get("deploy_history", []),
            incident_timeline=obs_data.get("incident_timeline", []),
            resolution_status=obs_data.get("resolution_status", "open"),
            runbook_draft=obs_data.get("runbook_draft", ""),
            comms_draft=obs_data.get("comms_draft", ""),
            last_action_result=obs_data.get("last_action_result", ""),
            done=payload.get("done", False),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
