from unitxt import add_to_catalog
from unitxt.metrics import RISK_TYPE_TO_CLASS, GraniteGuardianBase

for risk_type, risk_names in GraniteGuardianBase.available_risks.items():
    for risk_name in risk_names:
        metric_name = f"""granite_guardian.{risk_type.value}.{risk_name}"""
        metric = RISK_TYPE_TO_CLASS[risk_type](risk_name=risk_name)
        add_to_catalog(metric, name=f"metrics.{metric_name}", overwrite=True)
