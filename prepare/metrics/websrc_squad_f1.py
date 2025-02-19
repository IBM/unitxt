from unitxt import add_to_catalog
from unitxt.metrics import WebsrcSquadF1
from unitxt.test_utils.metrics import test_metric

metric = WebsrcSquadF1(n_resamples=None)

predictions = ["The 2nd", "The 1st"]
references = [["The 2nd"], ["The 2nd"]]

# how to create a metric which isn't updated in every sample when using UNITXT?
instance_targets = [
    {
        "websrc_squad_f1": 1.0,
        "score": 1.0,
        "score_name": "websrc_squad_f1",
    },
    {
        "websrc_squad_f1": 0.5,
        "score": 0.5,
        "score_name": "websrc_squad_f1",
    },
]
global_target = {
    "num_of_instances": 2,
    "websrc_squad_f1": 0.75,
    "score": 0.75,
    "score_name": "websrc_squad_f1",
}
outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
    task_data=[{"domain": "movie"}, {"domain": "movie"}],
)
add_to_catalog(metric, "metrics.websrc_squad_f1", overwrite=True)
