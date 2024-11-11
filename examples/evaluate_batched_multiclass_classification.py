import re
from typing import Any, Dict, List, NewType, Tuple

from unitxt.api import evaluate, load_dataset
from unitxt.artifact import fetch_artifact
from unitxt.operators import CollateInstances, Copy, FieldOperator, Rename
from unitxt.processors import PostProcess
from unitxt.serializers import MultiTypeSerializer, SingleTypeSerializer
from unitxt.task import Task
from unitxt.templates import InputOutputTemplate
from unitxt.text_utils import print_dict
from unitxt.type_utils import register_type

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


ClassList = NewType("ClassList", List[str])
register_type(ClassList)


class ClassListSerializer(SingleTypeSerializer):
    serialized_type = ClassList

    def serialize(self, value: ClassList, instance: Dict[str, Any]) -> str:
        return ",".join([f"{v}" for i, v in enumerate(value)])


task = Task(
    __description__="This is a batched multi-class classification task, where multiple texts are classified to a given set of options simultenously.",
    input_fields={
        "texts": EnumeratedList,
        "text_type": str,
        "classes": ClassList,
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
    serializer=MultiTypeSerializer(
        serializers=[EnumeratedListSerializer(), ClassListSerializer()]
    ),
)

for batch_size in [1, 2, 5, 10]:
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
        format="formats.llama3_instruct",
        num_demos=2,
        demos_pool_size=15,
        loader_limit=1000,
    )

    test_dataset = dataset["test"]

    from unitxt.inference import IbmGenAiInferenceEngine

    model_name = "meta-llama/llama-3-8b-instruct"
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
            "references",
            "score",
        ],
    )
