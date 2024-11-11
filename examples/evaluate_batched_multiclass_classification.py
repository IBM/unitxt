import re
from typing import Any, Dict, List, NewType, Tuple

import pandas as pd
from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.artifact import fetch_artifact
from unitxt.formats import SystemFormat
from unitxt.operators import CollateInstances, Copy, FieldOperator, Rename
from unitxt.processors import PostProcess
from unitxt.serializers import MultiTypeSerializer, SingleTypeSerializer
from unitxt.task import Task
from unitxt.templates import InputOutputTemplate
from unitxt.text_utils import print_dict
from unitxt.type_utils import register_type

logger = get_logger()


# Parse string in the format
# """1. class1
# 2. class2
# 3. class3"""
# to lis of tuples [ ("1", "class1"), ("2","class2"), ("3", "class3")]


class ParseEnumeratedList(FieldOperator):
    def process_value(self, text: Any) -> Any:
        result = []
        for x in text.split("\n"):
            line_result = re.findall(r"(\d+)\.\s*(\w+)", x)
            if len(line_result) == 1:
                result.append(line_result[0])
        return result


EnumeratedList = NewType("EnumeratedList", List[str])
register_type(EnumeratedList)


class EnumeratedListSerializer(SingleTypeSerializer):
    serialized_type = EnumeratedList

    def serialize(self, value: EnumeratedList, instance: Dict[str, Any]) -> str:
        return "\n".join([f"{i+1}. {v}" for i, v in enumerate(value)])


task = Task(
    __description__="This is a batched multi-class classification task, where multiple texts are classified to a given set of options simultenously.",
    input_fields={
        "texts": EnumeratedList,
        "text_type": str,
        "classes": EnumeratedList,
        "type_of_class": str,
    },
    reference_fields={"labels": EnumeratedList},
    prediction_type=List[Tuple[str, str]],
    metrics=["metrics.ner"],
    augmentable_inputs=["texts"],
    defaults={"text_type": "text"},
)

template = InputOutputTemplate(
    input_format="Classify each of the texts to its corresponding {type_of_class} from one of these options:\n{classes}\nReturn for each index the correspond class in a separate line.\nTexts:\n{texts}",
    target_prefix="Answer:\n",
    output_format="{labels}",
    postprocessors=[PostProcess(ParseEnumeratedList())],
    serializer=MultiTypeSerializer(serializers=[EnumeratedListSerializer()]),
)
df = pd.DataFrame(
    columns=["model", "batch_size", "num_instances", "f1_micro", "ci_low", "ci_high"]
)

for model_name in [
    "ibm/granite-8b-instruct-preview-4k",
    "meta-llama/llama-3-8b-instruct",
]:
    if model_name.startswith("ibm"):
        format = SystemFormat(
            demo_format=(
                "{instruction}\\N{source}\\N<|end_of_text|>\n"
                "<|start_of_role|>assistant<|end_of_role|>{target}\\N<|end_of_text|>\n"
                "<|start_of_role|>user<|end_of_role|>"
            ),
            model_input_format=(
                "<|start_of_role|>system<|end_of_role|>{system_prompt}<|end_of_text|>\n"
                "<|start_of_role|>user<|end_of_role|>{demos}{instruction}\\N{source}\\N<|end_of_text|>\n"
                "<|start_of_role|>assistant<|end_of_role|>"
            ),
        )
        batch_sizes = [50, 30, 10, 1]

    if model_name.startswith("meta-llama"):
        format = "formats.llama3_instruct"
        batch_sizes = [100, 50, 10, 1]

    for batch_size in batch_sizes:
        card, _ = fetch_artifact("cards.sst2")
        card.preprocess_steps.extend(
            [
                CollateInstances(batch_size=batch_size),
                Rename(field_to_field={"text": "texts", "label": "labels"}),
                Copy(field="text_type/0", to_field="text_type"),
                Copy(field="classes/0", to_field="classes"),
                Copy(
                    field="data_classification_policy/0",
                    to_field="data_classification_policy",
                ),
                Copy(field="type_of_class/0", to_field="type_of_class"),
            ]
        )
        card.task = task
        card.templates = [template]

        dataset = load_dataset(
            card=card,
            template_card_index=0,
            format=format,
            num_demos=1,
            demos_pool_size=5,
            loader_limit=10000,
            max_test_instances=1000 / batch_size,
        )

        test_dataset = dataset["test"]

        from unitxt.inference import IbmGenAiInferenceEngine

        inference_model = IbmGenAiInferenceEngine(
            model_name=model_name, max_new_tokens=1024
        )
        predictions = inference_model.infer(test_dataset)

        evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

        # Print results
        print_dict(
            evaluated_dataset[0],
            keys_to_print=[
                "source",
                "prediction",
                "processed_prediction",
                "processed_references",
            ],
        )

        global_scores = evaluated_dataset[0]["score"]["global"]
        df.loc[len(df)] = [
            model_name,
            batch_size,
            global_scores["num_of_instances"],
            global_scores["score"],
            global_scores["score_ci_low"],
            global_scores["score_ci_high"],
        ]

        df = df.round(decimals=2)
        logger.info(df.to_markdown())
