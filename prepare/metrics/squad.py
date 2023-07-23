from src.unitxt.catalog import add_to_catalog
from src.unitxt.metrics import Squad

metric = Squad()
add_to_catalog(metric, 'metrics.squad', overwrite=True)


