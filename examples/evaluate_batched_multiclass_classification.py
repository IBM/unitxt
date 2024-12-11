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
            line_result = re.findall(r"(\d+)\.\s*(.*)", x)
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
    __description__="""This is a batched multi-class classification task, where multiple 'texts' are classified to a given set of 'classes' in one inference call.
    The `type_of_class` field defines the type of classiication (e.g. "sentiment", "emotion", "topic" ) """,
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
    postprocessors=["processors.lower_case", PostProcess(ParseEnumeratedList())],
    serializer=MultiTypeSerializer(serializers=[EnumeratedListSerializer()]),
)
df = pd.DataFrame(
    columns=[
        "provider",
        "model",
        "batch_size",
        "num_instances",
        "f1_micro",
        "ci_low",
        "ci_high",
        "hellucinations",
    ]
)

for provider in [
    "watsonx",
    "bam",
]:
    for model_name in [
        "granite-3-8b-instruct",
        "llama-3-8b-instruct",
    ]:
        batch_sizes = [30, 20, 10, 5, 1]

        for batch_size in batch_sizes:
            card, _ = fetch_artifact("cards.banking77")
            card.preprocess_steps.extend(
                [
                    CollateInstances(batch_size=batch_size),
                    Rename(field_to_field={"text": "texts", "label": "labels"}),
                    Copy(field="text_type/0", to_field="text_type"),
                    Copy(field="classes/0", to_field="classes"),
                    Copy(field="type_of_class/0", to_field="type_of_class"),
                ]
            )
            card.task = task
            card.templates = [template]
            format = "formats.chat_api"
            if provider == "bam" and model_name.startswith("llama"):
                format = "formats.llama3_instruct"
            if provider == "bam" and model_name.startswith("granite"):
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

            dataset = load_dataset(
                card=card,
                template_card_index=0,
                format=format,
                num_demos=1,
                demos_pool_size=5,
                loader_limit=1000,
                max_test_instances=200 / batch_size,
            )

            test_dataset = dataset["test"]
            from unitxt.inference import CrossProviderInferenceEngine

            inference_model = CrossProviderInferenceEngine(
                model=model_name, max_tokens=1024, provider=provider
            )
            """
            We are using a CrossProviderInferenceEngine inference engine that supply api access to provider such as:
            watsonx, bam, openai, azure, aws and more.

            For the arguments these inference engines can receive, please refer to the classes documentation or read
            about the the open ai api arguments the CrossProviderInferenceEngine follows.
            """
            predictions = inference_model.infer(test_dataset)

            results = evaluate(predictions=predictions, data=test_dataset)

            print_dict(
                results.instance_scores[0],
                keys_to_print=[
                    "source",
                    "prediction",
                    "processed_prediction",
                    "processed_references",
                ],
            )

            global_scores = results.global_scores
            df.loc[len(df)] = [
                provider,
                model_name,
                batch_size,
                global_scores["num_of_instances"],
                global_scores["score"],
                global_scores["score_ci_low"],
                global_scores["score_ci_high"],
                1.0 - global_scores["in_classes_support"],
            ]

            df = df.round(decimals=2)
            logger.info(df.to_markdown())
