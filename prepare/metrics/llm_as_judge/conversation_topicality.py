import json

from unitxt import add_to_catalog
from unitxt.inference_engines import (
    IbmGenAiInferenceEngine,
    IbmGenAiInferenceEngineParams,
)
from unitxt.llm_as_judge import LLMAsJudge
from unitxt.metrics import (
    RandomForestMetricsEnsemble,
)

platform = "ibm_gen_ai"
gen_params = IbmGenAiInferenceEngineParams(max_new_tokens=256)

config_filepath = "prepare/metrics/llm_as_judge/ensemble_topicality_v1.json"

with open(config_filepath) as file:
    config = json.load(file)

metric_version_models = config["judges"]

template_lst = [sublist[0] for sublist in metric_version_models]
model_lst = [sublist[1] for sublist in metric_version_models]

inference_model_lst = []
for model_name in model_lst:
    inference_model = IbmGenAiInferenceEngine(
        model_name=model_name, parameters=gen_params
    )
    inference_model_lst.append(inference_model)

metric_lst = []
for i in range(len(model_lst)):
    model_label = model_lst[i].split("/")[1].replace("-", "")
    template_label = template_lst[i].split(".")[-1]
    metric_label = (
        "metrics.llm_as_judge.rating." + model_label + "_template_" + template_label
    )

    cur_metric = LLMAsJudge(
        inference_model=inference_model_lst[i],
        template=template_lst[i],
        task="rating.single_turn",
        main_score=metric_label,
        prediction_type="str",
    )
    metric_lst.append(cur_metric)

template_lst = [sublist[0] for sublist in metric_version_models]
model_lst = [sublist[1] for sublist in metric_version_models]


weights = RandomForestMetricsEnsemble.load_weights(config_filepath)
ensemble_metric = RandomForestMetricsEnsemble(
    metrics=metric_lst,
    weights=weights,
    __description__="A pre-trained Random Forest-based ensemble judge to assesss multi-turn conversation quality on Answer topicality: Response of the model only contains information that is related to and helpful for the user inquiry.",
)
# ensemble_metric.load_weights(config_filepath)
add_to_catalog(
    ensemble_metric,
    "metrics.llm_as_judge.conversation_answer_topicality.ensemble_v1_ibmgenai_judges",
    overwrite=True,
)
