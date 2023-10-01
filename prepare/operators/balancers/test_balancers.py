from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import DeterministicBalancer

balancer = DeterministicBalancer(fields=["target"], streams=["train", "validation", "test"])

add_to_catalog(balancer, "operators.balancers.outputs_balancer")

balancer = DeterministicBalancer(fields=["target"], streams=["validation", "test"])

add_to_catalog(balancer, "operators.balancers.test_outputs_balancer")
