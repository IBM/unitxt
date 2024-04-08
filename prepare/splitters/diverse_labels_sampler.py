from unitxt import add_to_catalog
from unitxt.splitters import DiverseLabelsSampler

add_to_catalog(
    DiverseLabelsSampler(),
    "splitters.diverse_labels_sampler",
    overwrite=True,
)
