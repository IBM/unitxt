from src.unitxt.blocks import (
    AddFields,
    FormTask,
    LoadHF,
    MapInstanceValues,
    RenameFields,
    TaskCard,
)
from src.unitxt.logging import get_logger
from src.unitxt.operators import (
    ApplyOperatorsField,
    CopyFields,
    GlobalMetricOnly,
    StreamRefiner,
)
from src.unitxt.standard import NaiveRecipe
from src.unitxt.text_utils import print_dict

logger = get_logger()

card = TaskCard(
    loader=LoadHF(path="glue", name="cola"),
    preprocess_steps=[
        StreamRefiner(max_instances=15),
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
        CopyFields(field="target", to_field="prediction"),
        AddFields(
            fields={
                "postprocessors": [
                    "processors.take_first_non_empty_line",
                    "processors.lower_case_till_punc",
                ],
            }
        ),
        ApplyOperatorsField(
            inputs_fields=["prediction", "references"],
            fields_to_treat_as_list=["references"],
            operators_field="postprocessors",
            default_operators=["processors.to_string_stripped"],
        ),
        GlobalMetricOnly(
            global_metric_name="matthews_correlation",
            local_scorer_name="lambda r,p: r==p",
            lambda_label_mapper='lambda label: 1 if label == "acceptable" else 0',
            wrapper_of_args_for_global_metric='{"references": refs, "predictions": preds}',
            is_global_cooked_from_locals=False,
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
