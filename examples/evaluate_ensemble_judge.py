import json

from unitxt import add_to_catalog, get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.blocks import Task, TaskCard
from unitxt.inference import (
    IbmGenAiInferenceEngine,
    IbmGenAiInferenceEngineParams,
)
from unitxt.llm_as_judge import LLMAsJudge
from unitxt.loaders import LoadFromDictionary
from unitxt.metrics import (
    RandomForestMetricsEnsemble,
)
from unitxt.operator import SequentialOperator
from unitxt.operators import CastFields
from unitxt.templates import InputOutputTemplate, TemplatesDict
from unitxt.text_utils import print_dict

logger = get_logger()

platform = "ibm_gen_ai"
gen_params = IbmGenAiInferenceEngineParams(max_new_tokens=256)

data = {
    "test": [
        {"conversation": "user: is eugene from walking dead really a scientist"},
        {"conversation": "user: is eugene from walking dead really a scientist"},
        {"conversation": "user: is eugene from walking dead really a scientist"},
    ]
}

predictions = [
    "In the Walking Dead series, Eugene Porter initially claims to be a scientist who knows the cure to the zombie plague. However, it is eventually revealed that he is not a scientist, but a high school science teacher who lied about knowing the cure to manipulate other survivors into helping him.",
    "No, Eugene Porter, played by Josh McDermitt, is not a scientist in real life. In the Walking Dead TV series, he portrays a character who initially appears to be a scientist with knowledge of a cure for the zombie outbreak. However, it is later revealed that he is actually a high school science teacher who lied about his credentials to manipulate other survivors into taking him to Washington D.C.",
    "thank you for the question.",
]

add_to_catalog(
    SequentialOperator(
        steps=[
            CastFields(
                fields={"prediction": "float"},
                failure_defaults={"prediction": 0.5},
            ),
            CastFields(
                fields={"references": "float"},
                process_every_value=True,
            ),
        ]
    ),
    "processors.cast_to_float_return_0_5_if_failed",
    overwrite=True,
)

config_filepath = (
    "prepare/templates/response_assessment/judges/ensemble_related_v1.json"
)
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

weights = RandomForestMetricsEnsemble.load_weights(config_filepath)
ensemble_metric = RandomForestMetricsEnsemble(metrics=metric_lst, weights=weights)
# ensemble_metric.load_weights(config_filepath)
add_to_catalog(
    ensemble_metric,
    "metrics.llm_as_judge.related.ensemble_v1_ibmgenai_judges",
    overwrite=True,
)

card = TaskCard(
    loader=LoadFromDictionary(data=data),
    task=Task(
        input_fields={"conversation": "str"},
        reference_fields={},
        prediction_type="str",
        # metrics=[ensemble_metric],
        metrics=["metrics.llm_as_judge.related.ensemble_v1_ibmgenai_judges"],
    ),
    templates=TemplatesDict(
        {
            "simple": InputOutputTemplate(
                input_format="{conversation}",
                output_format="",
            )
        }
    ),
)

test_dataset = load_dataset(card=card, template_card_index="simple")["test"]
evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

for instance in evaluated_dataset:
    print_dict(
        instance,
        keys_to_print=[
            "source",
            "prediction",
            "processed_prediction",
            "references",
            "score",
        ],
    )
