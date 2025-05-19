
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
    templates=["templates.empty[postprocessors=[processors.cast_to_float_return_nan_if_failed]]"]
)

test_card(card, demos_taken_from="test", strict=False)

add_to_catalog(
    card,
    "cards.judege_bench.toxic_chat.toxicity",
    overwrite=True,
)
