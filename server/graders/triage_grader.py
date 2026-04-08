from typing import List, Dict

def grade_triage(agent_triages: List[Dict], ground_truth: List[Dict]) -> float:
    if not agent_triages or not ground_truth:
        return 0.0
        
    severity_rank = ['P0', 'P1', 'P2']
    
    severity_score = 0.0
    team_score = 0.0
    
    # We map agent triages to ground truth by alert_id to avoid ordering issues
    gt_map = {g['alert_id']: g for g in ground_truth if 'true_severity' in g}
    
    for a in agent_triages:
        alert_id = a.get('alert_id')
        if alert_id in gt_map:
            gt = gt_map[alert_id]
            
            # Severity
            ag_sev = a.get('severity')
            gt_sev = gt['true_severity']
            
            if ag_sev == gt_sev:
                severity_score += 1.0
            else:
                try:
                    if abs(severity_rank.index(ag_sev) - severity_rank.index(gt_sev)) == 1:
                        severity_score += 0.5
                except ValueError:
                    pass
                    
            # Team
            if a.get('team') == gt['true_team']:
                team_score += 1.0
                
    severity_score /= len(gt_map)
    team_score /= len(gt_map)
    
    return round(0.5 * severity_score + 0.5 * team_score, 4)
