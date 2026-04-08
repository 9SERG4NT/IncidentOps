from typing import Dict, Any

def grade_rca(hypothesis: str, scenario_id: str, root_cause_keywords: Dict[str, Any]) -> float:
    if not hypothesis or not root_cause_keywords:
        return 0.0
        
    kw = root_cause_keywords
    h = hypothesis.lower()
    
    wrong_hits = sum(1 for w in kw.get('wrong', []) if w.lower() in h)
    if wrong_hits > 0:
        return 0.0
        
    req_list = kw.get('required', [])
    sup_list = kw.get('supporting', [])
    
    if not req_list:
        return 1.0
        
    req_hits = sum(1 for r in req_list if r.lower() in h)
    sup_hits = sum(1 for s in sup_list if s.lower() in h)
    
    base = req_hits / len(req_list)
    bonus = 0.0
    if sup_list:
        bonus = min(sup_hits / len(sup_list) * 0.2, 0.2)
        
    return round(min(base + bonus, 1.0), 4)
