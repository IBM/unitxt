from unitxt import add_to_catalog
from unitxt.metrics import GraniteGuardianWMLMetric, MetricPipeline
from unitxt.operators import Set

for risk_type, risk_names in GraniteGuardianWMLMetric.available_risks.items():
    for risk_name in risk_names:
        metric_name = f"""granite_guardian.{risk_type.value}.{risk_name}"""
        metric = GraniteGuardianWMLMetric(
            main_score=metric_name, risk_name=risk_name, risk_type=risk_type.name
        )

        metric_pipeline = MetricPipeline(
            main_score=metric_name,
            metric=metric,
            preprocess_steps=[
                Set(
                    fields={
                        "prediction": 0.0,  # these are not used by the metric
                        "references": [0.0],
                    }
                ),
            ],
        )

        add_to_catalog(metric_pipeline, name=f"metrics.{metric_name}", overwrite=True)
