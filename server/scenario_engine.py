import json
import random
from typing import Dict, List, Any
from pathlib import Path
from server.models import Alert, LogEntry, MetricSeries, DeployEvent

class ScenarioEngine:
    def __init__(self):
        self.templates_dir = Path(__file__).parent.parent / 'scenarios'
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, Dict]:
        templates = {}
        for f in self.templates_dir.glob('*.json'):
            with open(f) as file:
                templates[f.stem] = json.load(file)
        return templates

    def generate(self, template_id: str, seed: int) -> Dict[str, Any]:
        rng = random.Random(seed)
        if template_id not in self.templates:
            raise ValueError(f"Scenario template {template_id} not found.")
        template = self.templates[template_id]

        # Generate base timeline (60 minutes of logs/metrics)
        timeline = self._generate_timeline(rng)

        # Inject root cause anomaly
        if 'root_anomaly' in template:
            root_anomaly = template['root_anomaly']
            self._inject_root_anomaly(timeline, root_anomaly, rng)

        # Cascade to dependents
        for dep in template.get('dependents', []):
            lag = rng.randint(2, 8)
            self._cascade_to_dependent(timeline, dep, lag, rng)

        # Generate logs from anomalies
        logs = self._generate_logs(timeline, template, rng)

        # Build final realistic alert list with noise
        base_alerts = template.get('real_alerts', [])
        noise_alerts = self._generate_noise_alerts(rng.randint(4, 7), rng)
        all_alerts = base_alerts + noise_alerts
        rng.shuffle(all_alerts)
        
        # Deploy history
        deploys = self._generate_deploy_history(template, rng)

        return {
            'id': template_id,
            'seed': seed,
            'alerts': all_alerts,
            'logs': logs,
            'metrics': timeline['metrics'],
            'deploy_history': deploys,
            'root_cause_keywords': template.get('root_cause_keywords', {}),
            'valid_fixes': template.get('valid_fixes', []),
            'relevant_services': template.get('relevant_services', []),
        }

    def _generate_timeline(self, rng: random.Random) -> Dict:
        timestamps = [f"2023-10-01T00:{m:02d}:00Z" for m in range(60)]
        metrics = {}
        for service in ['payment-api', 'order-service', 'db-cluster']:
            metrics[f"{service}.cpu_percent"] = MetricSeries(
                name=f"{service}.cpu_percent", unit="%", timestamps=timestamps,
                values=[rng.uniform(20, 40) for _ in timestamps]
            )
            metrics[f"{service}.memory_percent"] = MetricSeries(
                name=f"{service}.memory_percent", unit="%", timestamps=timestamps,
                values=[rng.uniform(40, 60) for _ in timestamps]
            )
            metrics[f"{service}.requests_5xx"] = MetricSeries(
                name=f"{service}.requests_5xx", unit="ops", timestamps=timestamps,
                values=[rng.uniform(0, 0.5) for _ in timestamps]
            )
        return {'timestamps': timestamps, 'metrics': metrics}

    def _inject_root_anomaly(self, timeline: Dict, anomaly: Dict, rng: random.Random):
        start_min = anomaly.get('start_min', 30)
        end_min = anomaly.get('end_min', 60)
        metric_name = anomaly.get('metric')
        if metric_name in timeline['metrics']:
            for i in range(start_min, end_min):
                timeline['metrics'][metric_name].values[i] += rng.uniform(40, 60)

    def _cascade_to_dependent(self, timeline: Dict, dep: Dict, lag: int, rng: random.Random):
        metric_name = dep.get('metric')
        # Simple cascade logic at start+lag
        if metric_name in timeline['metrics']:
            for i in range(30 + lag, 60):
                timeline['metrics'][metric_name].values[i] += rng.uniform(5, 15) * dep.get('cascade_factor', 1)

    def _generate_logs(self, timeline: Dict, template: Dict, rng: random.Random) -> List[Dict]:
        logs = []
        for i, ts in enumerate(timeline['timestamps']):
            for srv in ['payment-api', 'order-service', 'db-cluster']:
                # Background info logs
                if rng.random() > 0.5:
                    logs.append({"ts": ts, "level": "INFO", "service": srv, "message": f"Processed request successfully for {srv}"})
                
                # If anomaly is happening, inject errors
                if i >= 30: # Hardcoded anomaly window for simplicity
                    if srv in template.get('relevant_services', []):
                        if rng.random() > 0.7:
                            logs.append({"ts": ts, "level": "ERROR", "service": srv, "message": f"Connection pool exhausted or memory high in {srv}"})
        return logs

    def _generate_noise_alerts(self, count: int, rng: random.Random) -> List[Dict]:
        noise = []
        for i in range(count):
            noise.append({
                "alert_id": f"noise_{i}",
                "title": f"Intermittent latency in {rng.choice(['frontend', 'worker-queue'])}",
                "service": rng.choice(['frontend', 'worker-queue']),
                "timestamp": f"2023-10-01T00:{rng.randint(0, 59):02d}:00Z",
                "raw_value": rng.uniform(500, 800),
                "threshold": 500,
                # Intentionally missing true_severity and true_team or set to false flags for grading
            })
        return noise

    def _generate_deploy_history(self, template: Dict, rng: random.Random) -> List[Dict]:
        deploys = []
        services = ['payment-api', 'order-service', 'frontend']
        for i in range(10):
            srv = rng.choice(services)
            deploys.append({
                "deploy_id": f"dep_{i}",
                "service": srv,
                "sha": f"{rng.randint(100000, 999999)}",
                "timestamp": f"2023-10-01T00:{rng.randint(0, 59):02d}:00Z",
                "deployer": f"user_{rng.randint(1, 5)}",
                "diff_summary": "Minor fixes and updates"
            })
        # Inject bad deploy if missing
        if 'root_cause_keywords' in template and 'v2.4.1' in template['root_cause_keywords'].get('required', []):
            deploys.append({
                "deploy_id": "dep_bad",
                "service": "payment-api",
                "sha": "bad123",
                "timestamp": "2023-10-01T00:25:00Z",
                "deployer": "user_1",
                "diff_summary": "Update payment-api to v2.4.1"
            })
        return sorted(deploys, key=lambda x: x['timestamp'])
