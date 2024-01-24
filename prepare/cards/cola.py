from src.unitxt.blocks import (
    AddFields,
    FormTask,
    LoadHF,
    MapInstanceValues,
    RenameFields,
    TaskCard,
)
from src.unitxt.logging_utils import get_logger
from src.unitxt.operators import (
    ApplyOperatorsField,
    GlobalClassificationMetric,
    Perturbate,
    StreamRefiner,
)
from src.unitxt.standard import NaiveRecipe
from src.unitxt.text_utils import print_dict

logger = get_logger()

card = TaskCard(
    loader=LoadHF(path="glue", name="cola"),
    preprocess_steps=[
        StreamRefiner(max_instances=200),
        "splitters.small_no_test",
        MapInstanceValues(mappers={"label": {"0": "unacceptable", "1": "acceptable"}}),
        RenameFields(field_to_field={"sentence": "text"}),
        AddFields(
            fields={
                "classes": ["unacceptable", "acceptable"],
                "text_type": "text",
                "type_of_class": "grammatical acceptability",
            }
        ),
        FormTask(
            inputs=["text", "text_type", "classes", "type_of_class"],
            outputs=["label"],
        ),
        "templates.classification.multi_class.default",
        Perturbate(
            field="target",
            to_field="prediction",
            select_from=["unacceptable", "acceptable"],
            percentage_to_perturbate=30,
        ),
        AddFields(
            fields={
                "postprocessors": [
                    "processors.take_first_non_empty_line",
                    "processors.lower_case_till_punc",
                ],
            }
        ),
        ApplyOperatorsField(
            operators_field="postprocessors",
            default_operators=["processors.to_string_stripped"],
        ),
        GlobalClassificationMetric(
            pred_field_name="prediction",
            refs_field_name="references",
            global_metric_names=[
                "metrics.matthews_correlation",
                "metrics.f1_micro",
                "metrics.f1_macro",
                "metrics.accuracy",
            ],
        ),
    ],
)

naive_recipe = NaiveRecipe(card=card)

multi_stream = naive_recipe()

for stream_name, stream in multi_stream.items():
    logger.info(f"Stream: {stream_name}")
    for instance in stream:
        print_dict(instance)


# test_card(card, strict=False)
# add_to_catalog(card, "cards.cola_new", overwrite=True)
