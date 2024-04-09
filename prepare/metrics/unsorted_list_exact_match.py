from unitxt import add_to_catalog
from unitxt.metrics import UnsortedListExactMatch

metric = UnsortedListExactMatch()

add_to_catalog(metric, "metrics.unsorted_list_exact_match", overwrite=True)
