from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import DeterministicBalancer

balancer = DeterministicBalancer(fields=["target"])

add_to_catalog(balancer, "operators.balancers.outputs_balancer", overwrite=True)

balancer = DeterministicBalancer(fields=["target"], apply_to_streams=["test"])

add_to_catalog(balancer, "operators.balancers.test_outputs_balancer", overwrite=True)

balancer = DeterministicBalancer(fields=["target"], apply_to_streams=["validation", "test"])

add_to_catalog(balancer, "operators.balancers.test_validation_outputs_balancer", overwrite=True)
