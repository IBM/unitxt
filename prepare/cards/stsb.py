from src.unitxt.blocks import (
    FormTask,
    LoadHF,
    OutputQuantizingTemplate,
    SplitRandomMix,
    TaskCard,
)
from src.unitxt.logging_utils import get_logger
from src.unitxt.operators import (
    ExecuteQuery,
    GlobalClassificationMetricOnly,
    Perturbate,
    StreamRefiner,
)
from src.unitxt.standard import NaiveRecipe
from src.unitxt.text_utils import print_dict

logger = get_logger()

card = TaskCard(
    loader=LoadHF(path="glue", name="stsb"),
    preprocess_steps=[
        StreamRefiner(max_instances=200),
        SplitRandomMix(
            {"train": "train[95%]", "validation": "train[5%]", "test": "validation"}
        ),
        FormTask(
            inputs=["sentence1", "sentence2"],
            outputs=["label"],
            metrics=["metrics.pearson", "metrics.spearman"],
        ),
        OutputQuantizingTemplate(
            input_format="""
               Given this sentence: '{sentence1}', on a scale of 1 to 5, how similar in meaning is it to this sentence: '{sentence2}'?
            """.strip(),
            output_format="{label}",
            quantum=0.2,
        ),
        # round float numbers just for the fun
        ExecuteQuery(
            query="[int(0.5 + float(v)) for v in references]", to_field="references"
        ),
        ExecuteQuery(query="int(0.5 + float(target))", to_field="target"),
        Perturbate(
            field="target",
            to_field="prediction",
            percentage_to_perturbate=30,
        ),
        GlobalClassificationMetricOnly(
            pred_field_name="prediction",
            refs_field_name="references",
            global_metric_names=["metrics.pearson", "metrics.spearman"],
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
# add_to_catalog(card, "cards.stsb", overwrite=True)
