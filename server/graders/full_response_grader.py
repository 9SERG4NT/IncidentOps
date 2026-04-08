from server.graders.triage_grader import grade_triage
from server.graders.rca_grader import grade_rca
from typing import Any

def grade_full_response(env: Any) -> float:
    total_real = [a for a in env.scenario.get('alerts', []) if 'true_severity' in a]
    triage_score = grade_triage(env.agent_triages, total_real)
    
    rca_score = grade_rca(env.agent_hypothesis or '', env.scenario.get('id', ''), env.scenario.get('root_cause_keywords', {}))
    
    fix_score = 1.0 if env.agent_fix in env.scenario.get('valid_fixes', []) else 0.0
    
    runbook_score = _grade_runbook(env.state_data.get('runbook_draft', ''))
    comms_score = _grade_comms(env.state_data.get('comms_draft', ''))
    
    # Penalize redundant queries
    timeline = env.state_data.get('incident_timeline', [])
    query_actions = [a for a in timeline if 'query_logs' in a]
    redundant = max(0, len(query_actions) - len(set(query_actions)))
    penalty = 0.05 * redundant
    
    total = 0.20 * triage_score + 0.35 * rca_score + 0.20 * fix_score + 0.15 * runbook_score + 0.10 * comms_score - penalty
    return max(0.0, min(total, 1.0))

def _grade_runbook(text: str) -> float:
    if not text:
        return 0.0
    sections = ['symptoms', 'root cause', 'fix', 'prevention']
    hits = sum(1 for s in sections if s in text.lower())
    length_ok = 1.0 if 100 <= len(text.split()) <= 800 else 0.5
    return round((hits / len(sections)) * length_ok, 4)

def _grade_comms(text: str) -> float:
    if not text:
        return 0.0
    length_ok = 1.0 if 20 <= len(text.split()) <= 200 else 0.5
    has_target = 'resolved' in text.lower() or 'monitoring' in text.lower() or 'mitigated' in text.lower()
    return round((0.5 if has_target else 0.0) + (0.5 * length_ok), 4)
