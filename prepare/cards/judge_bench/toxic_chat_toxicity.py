
from unitxt.api import load_dataset
from unitxt.blocks import (
    MapInstanceValues,
    Rename,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.llm_as_judge_constants import DirectCriteriaCatalogEnum
from unitxt.loaders import LoadJsonFile
from unitxt.operators import Copy
from unitxt.task import Task
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadJsonFile(
        files={
            "train": "https://raw.githubusercontent.com/dmg-illc/JUDGE-BENCH/refs/heads/master/data/toxic_chat/toxic_chat_train.json",
            "test":"https://raw.githubusercontent.com/dmg-illc/JUDGE-BENCH/refs/heads/master/data/toxic_chat/toxic_chat_test.json"
        },
        data_classification_policy=["public"],
        data_field="instances",
    ),
    preprocess_steps=[
        Rename(field="instance", to_field="text"),
        Rename(field="annotations/toxicity/majority_human", to_field="label"),
        MapInstanceValues(mappers={
            "label": {
                "0": "No",
                "1": "Yes"
            },
        }),
        Copy(field="label", to_field="label_value"),
        MapInstanceValues(mappers={
            "label_value": DirectCriteriaCatalogEnum.TOXICITY.value.option_map,
        }),
    ],
    task=Task(
        input_fields={"text": str, "label": str},
        reference_fields={"label_value": float},
        prediction_type=float,
        metrics=[
            "metrics.spearman",
            "metrics.accuracy"
        ],
        default_template="templates.empty[postprocessors=[processors.cast_to_float_return_nan_if_failed]]"
    ),
    templates=[]
)

dataset = load_dataset(
    card=card,
    split="test")


test_card(card, demos_taken_from="test", strict=False, loader_limit=100)
add_to_catalog(
    card,
    "cards.judege_bench.toxic_chat.toxicity",
    overwrite=True,
)

# params = f"[criteria=metrics.llm_as_judge.direct.criteria.toxicity,context_fields=[],check_positional_bias=False]"

# metric_inference_engine = MetricInferenceEngine(
#     metric=f"metrics.llm_as_judge.direct.watsonx.llama3_3_70b{params}",
#     prediction_field="text",
# )
# predictions = [p["toxicity"] for p in metric_inference_engine.infer(dataset)]

# results = evaluate(predictions=predictions, data=dataset)
# parsed_results = {"spearmanr": results.global_scores["spearmanr"], "accuracy": results.global_scores["accuracy"]}
# print(parsed_results)
