from src.unitxt import add_to_catalog
from src.unitxt.metrics import UnsortedListExactMatch

metric = UnsortedListExactMatch()

add_to_catalog(metric, "metrics.unsorted_list_exact_match", overwrite=True)
