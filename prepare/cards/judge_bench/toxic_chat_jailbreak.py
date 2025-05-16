
from unitxt.api import load_dataset
from unitxt.blocks import (
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.llm_as_judge_constants import DirectCriteriaCatalogEnum
from unitxt.loaders import LoadFromAPI
from unitxt.operators import Copy, MapInstanceValues, Rename
from unitxt.splitters import SplitRandomMix
from unitxt.task import Task
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadFromAPI(
        urls={
            # "train": "https://raw.githubusercontent.com/dmg-illc/JUDGE-BENCH/refs/heads/master/data/toxic_chat/toxic_chat_train.json",
            "test":"https://raw.githubusercontent.com/dmg-illc/JUDGE-BENCH/refs/heads/master/data/toxic_chat/toxic_chat_test.json"
        },
        data_classification_policy=["public"],
        data_field="instances",
        loader_limit=10,
    ),
    preprocess_steps=[
        SplitRandomMix(
            mix={
                "test": "test[100%]",
            }
        ),
        Rename(field="instance", to_field="text"),
        Rename(field="annotations/jailbreaking/majority_human", to_field="label"),
        MapInstanceValues(mappers={
            "label": {
                "0": "No",
                "1": "Yes"
            },
        }),
        Copy(field_to_field={"label": "label_value"}),
        MapInstanceValues(mappers={
            "label_value": DirectCriteriaCatalogEnum.JAILBREAK_USER_MESSAGE.value.option_map,
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
    "cards.judege_bench.toxic_chat.jailbreaking",
    overwrite=True,
)

# params = f"[criteria=metrics.llm_as_judge.direct.criteria.user_message_jailbreak,context_fields=[],check_positional_bias=False]"

# metric_inference_engine = MetricInferenceEngine(
#     metric=f"metrics.llm_as_judge.direct.watsonx.llama3_3_70b{params}",
#     prediction_field="text",
# )
# predictions = [p["user_message_jailbreak"] for p in metric_inference_engine.infer(dataset)]

# results = evaluate(predictions=predictions, data=dataset)
# parsed_results = {"spearmanr": results.global_scores["spearmanr"], "accuracy": results.global_scores["accuracy"]}
# print(parsed_results)
