from unitxt.catalog import add_to_catalog
from unitxt.operators import (
    DeterministicBalancer,
    LengthBalancer,
    MinimumOneExamplePerLabelRefiner,
)

balancer = DeterministicBalancer(fields=["reference_fields/label"])

add_to_catalog(balancer, "operators.balancers.classification.by_label", overwrite=True)

balancer = DeterministicBalancer(fields=["reference_fields/answer"])

add_to_catalog(balancer, "operators.balancers.qa.by_answer", overwrite=True)

balancer = LengthBalancer(fields=["reference_fields/labels"], segments_boundaries=[1])

add_to_catalog(
    balancer, "operators.balancers.multi_label.zero_vs_many_labels", overwrite=True
)

balancer = LengthBalancer(fields=["reference_fields/labels"], segments_boundaries=[1])

add_to_catalog(
    balancer, "operators.balancers.ner.zero_vs_many_entities", overwrite=True
)

balancer = MinimumOneExamplePerLabelRefiner(fields=["reference_fields/label"])

add_to_catalog(
    balancer,
    "operators.balancers.classification.minimum_one_example_per_class",
    overwrite=True,
)
