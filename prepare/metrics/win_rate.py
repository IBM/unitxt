from unitxt.catalog import add_to_catalog
from unitxt.metrics import WeightedWinRate, WeightedWinRateCorrelation

metric = WeightedWinRateCorrelation()
add_to_catalog(metric, "metrics.weighted_win_rate_correlation", overwrite=True)

metric = WeightedWinRate()
add_to_catalog(metric, "metrics.weighted_win_rate", overwrite=True)
