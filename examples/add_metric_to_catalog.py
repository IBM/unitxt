import evaluate
import unitxt
from unitxt.blocks import ToString
from unitxt.catalog import add_to_catalog
from unitxt.metrics import Accuracy
from unitxt.text_utils import print_dict

add_to_catalog(ToString(), "processors.to_string", overwrite=True)
add_to_catalog(Accuracy(), "metrics.accuracy", overwrite=True)

data = [
    {
        "group": "group1",
        "references": ["333", "4"],
        "source": "source1",
        "target": "target1",
    },
    {"group": "group1", "references": ["4"], "source": "source2", "target": "target2"},
    {"group": "group2", "references": ["3"], "source": "source3", "target": "target3"},
    {"group": "group2", "references": ["3"], "source": "source4", "target": "target4"},
]

for d in data:
    d["task_data"] = {}
    d["metrics"] = ["metrics.accuracy"]
    d["postprocessors"] = ["processors.to_string"]

predictions = ["4", " 3", "3", "3"]

metric = evaluate.load(unitxt.metric_url)

results = metric.compute(predictions=predictions, references=data, flatten=True)

print_dict(results[0])
