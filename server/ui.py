import json
import gradio as gr
from pydantic import ValidationError

from server.models import ActionType, IncidentAction


def init_environment(task_name):
    from server.main import reset as api_reset, ResetRequest
    req = ResetRequest(task=task_name, seed=42)
    try:
        resp = api_reset(req)
        env_id = resp["env_id"]
        obs = resp["observation"]
        return env_id, obs, f"Environment Initialized (Task: {task_name}).\nSee observation below."
    except Exception as e:
        return "", {}, f"Error initializing environment: {e}"

def take_action(env_id, action_type_str, params_text):
    if not env_id:
        return {}, "Error: No active environment. Initialize first."
    
    try:
        params = json.loads(params_text) if params_text.strip() else {}
    except Exception as e:
        return {}, f"Error parsing parameters JSON:\n{e}"
        
    try:
        from server.main import StepRequest, step as api_step
        action = IncidentAction(action_type=ActionType(action_type_str), parameters=params)
        req = StepRequest(env_id=env_id, action=action)
    except ValidationError as ve:
        return {}, f"Validation Error in Action:\n{ve}"
    except Exception as e:
        return {}, f"Error creating action: {e}"
        
    try:
        resp = api_step(req)
        obs_json = resp.observation.model_dump()
        reward = resp.reward.model_dump()
        done = resp.done
        
        breakdown_str = ", ".join(f"{k}: {v:.3f}" for k, v in reward.get("breakdown", {}).items())
        status = (
            f"Action '{action_type_str}' executed.\n"
            f"Reward: {reward.get('value', 0):.4f}\n"
            f"Breakdown: {breakdown_str or 'none'}\n"
            f"Done: {done}"
        )
        return obs_json, status
    except Exception as e:
        return {}, f"Error executing action:\n{e}"

def inject_example_params(action_type_str):
    examples = {
        "triage_alert": '{\n  "alert_id": "alt-123",\n  "severity": "P1",\n  "team": "backend"\n}',
        "query_logs": '{\n  "service": "database",\n  "level": "ERROR"\n}',
        "query_metrics": '{\n  "service": "backend"\n}',
        "page_team": '{\n  "team": "infra"\n}',
        "hypothesize": '{\n  "hypothesis": "Database connection pool exhausted due to spike in frontend traffic."\n}',
        "apply_fix": '{\n  "fix_id": "fix_scale_db_pool"\n}',
        "write_runbook": '{\n  "text": "# Runbook\\nIncrease DB Max Connections if >90% saturated."\n}',
        "write_comms": '{\n  "text": "We are currently investigating elevated latency."\n}',
        "close_incident": '{}'
    }
    return examples.get(action_type_str, "{}")

def build_ui():
    with gr.Blocks(title="IncidentOps UI", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🚨 IncidentOps: Production Incident Response UI
            Welcome to the interactive interface for IncidentOps. Use this UI to step into the role of a Site Reliability Engineer (SRE).
            Select a task, initialize the environment, inspect the observation data, and execute actions to resolve the incident!
            """
        )
        
        env_id_state = gr.State("")
        
        with gr.Row():
            # LEFT CPNL - Controls
            with gr.Column(scale=1):
                gr.Markdown("### 1. Control Panel")
                from server.main import VALID_TASKS
                task_dropdown = gr.Dropdown(choices=list(VALID_TASKS), value="alert-triage", label="Select Task")
                init_btn = gr.Button("Initialize Environment", variant="primary")
                
                gr.Markdown("---")
                
                gr.Markdown("### 2. Take Action")
                action_dropdown = gr.Dropdown(choices=[e.value for e in ActionType], value=ActionType.TRIAGE_ALERT.value, label="Action Type")
                params_input = gr.Code(language="json", label="Action Parameters (JSON)", value=inject_example_params(ActionType.TRIAGE_ALERT.value), lines=6)
                
                # Auto-update the JSON parameter template when action type is changed
                action_dropdown.change(
                    fn=inject_example_params,
                    inputs=[action_dropdown],
                    outputs=[params_input]
                )
                
                action_btn = gr.Button("Submit Action", variant="primary")
                
                status_box = gr.Textbox(label="Status / Last Action Result", interactive=False, lines=6)
                
            # RIGHT CPNL - Observation State
            with gr.Column(scale=2):
                gr.Markdown("### 3. Current Observation State")
                observation_json = gr.JSON(label="Environment State", value={})
                
        # Wire up initialization
        init_btn.click(
            fn=init_environment,
            inputs=[task_dropdown],
            outputs=[env_id_state, observation_json, status_box]
        )
        
        # Wire up action submission
        action_btn.click(
            fn=take_action,
            inputs=[env_id_state, action_dropdown, params_input],
            outputs=[observation_json, status_box]
        )
        
    return demo
