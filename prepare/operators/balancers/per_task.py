from unitxt.catalog import add_to_catalog
from unitxt.operators import (
    DeterministicBalancer,
    LengthBalancer,
    MinimumOneExamplePerLabelRefiner,
)

balancer = DeterministicBalancer(fields=["outputs/label"])

add_to_catalog(balancer, "operators.balancers.classification.by_label", overwrite=True)

balancer = DeterministicBalancer(fields=["outputs/answer"])

add_to_catalog(balancer, "operators.balancers.qa.by_answer", overwrite=True)

balancer = LengthBalancer(fields=["outputs/labels"], segments_boundaries=[1])

add_to_catalog(
    balancer, "operators.balancers.multi_label.zero_vs_many_labels", overwrite=True
)

balancer = LengthBalancer(fields=["outputs/labels"], segments_boundaries=[1])

add_to_catalog(
    balancer, "operators.balancers.ner.zero_vs_many_entities", overwrite=True
)

balancer = MinimumOneExamplePerLabelRefiner(fields=["outputs/label"])

add_to_catalog(
    balancer,
    "operators.balancers.classification.minimum_one_example_per_class",
    overwrite=True,
)
