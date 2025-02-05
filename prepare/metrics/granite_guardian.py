from unitxt import add_to_catalog
from unitxt.metrics import GraniteGuardianWMLMetric

for risk_type, risk_names in GraniteGuardianWMLMetric.available_risks.items():
    for risk_name in risk_names:
        metric_name = f"""granite_guardian.{risk_type.value}.{risk_name}"""
        metric = GraniteGuardianWMLMetric(risk_name=risk_name, risk_type=risk_type.name)
        add_to_catalog(metric, name=f"metrics.{metric_name}", overwrite=True)
