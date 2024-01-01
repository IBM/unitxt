from src.unitxt import add_to_catalog
from src.unitxt.metrics import Perplexity
from src.unitxt.test_utils.metrics import test_metric

metric = Perplexity(model_name="google/flan-t5-small")

predictions = ["who are we?",
               "who are we?",
               "who are we?",
               "what are we saving?",
               "what are we saving?",
               "what are we saving?"]

references = [
    ["we are the world"],
    ["we are the children"],
    ["we are the ones"],
    ["we make a brighter day"],
    ["we are making a choice"],
    ["we are saving our own lives"]
]

instance_targets = [
    {"map": 0.42, "score": 0.42, "score_name": "map"},
    {"map": 1.0, "score": 1.0, "score_name": "map"},
    {"map": 0, "score": 0, "score_name": "map"},
    {"map": 0, "score": 0, "score_name": "map"},
    {"map": 0, "score": 0, "score_name": "map"},
]

global_target = {
    "map": 0.28,
    "score": 0.28,
    "score_name": "map",
    "map_ci_low": 0.0,
    "map_ci_high": 0.8,
    "score_ci_low": 0.0,
    "score_ci_high": 0.8,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.perplexity", overwrite=True)
