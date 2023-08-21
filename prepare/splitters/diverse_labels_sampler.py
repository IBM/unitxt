
from src.unitxt import add_to_catalog
from src.unitxt.splitters import DiverseLabelsSampler


add_to_catalog(
    DiverseLabelsSampler(),
    "splitters.diverse_labels_sampler",
    overwrite=True,
)


