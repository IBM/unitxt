from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import DeterministicBalancer

balancer = DeterministicBalancer(fields=["target"])

add_to_catalog(balancer, "operators.balancers.balanced_targets", overwrite=True)
