from typing import Any, Dict, List, Literal, NewType, TypedDict

from unitxt import load_dataset
from unitxt.card import TaskCard
from unitxt.loaders import LoadFromDictionary
from unitxt.serializers import SingleTypeSerializer
from unitxt.task import Task
from unitxt.templates import InputOutputTemplate
from unitxt.text_utils import print_dict
from unitxt.type_utils import register_type


class CustomTurn(TypedDict):
    role: Literal["system", "user", "agent"]
    content: str


CustomDialog = NewType("CustomDialog", List[CustomTurn])

register_type(CustomTurn)
register_type(CustomDialog)


class CustomDialogSerializer(SingleTypeSerializer):
    serialized_type = CustomDialog

    def serialize(self, value: CustomDialog, instance: Dict[str, Any]) -> str:
        return (
            "___\n"
            + "\n".join(f"  {turn['role']}: {turn['content']}" for turn in value)
            + "\n___"
        )


dialog_summarization_task = Task(
    input_fields={"dialog": CustomDialog},
    reference_fields={"summary": str},
    prediction_type=str,
    metrics=["metrics.rouge"],
)

data = {
    "test": [
        {
            "dialog": [
                {"role": "user", "content": "What is the time?"},
                {"role": "system", "content": "4:13 PM"},
            ],
            "summary": "User asked for the time and got an answer.",
        }
    ]
}


card = TaskCard(
    loader=LoadFromDictionary(data=data),
    task=dialog_summarization_task,
)


dataset = load_dataset(
    card=card,
    template=InputOutputTemplate(
        instruction="Summarize the following dialog.",
        input_format="{dialog}",
        output_format="{summary}",
    ),
    serializer=CustomDialogSerializer(),
)

print_dict(
    dataset["test"][0],
    keys_to_print=[
        "source",
        "target",
    ],
)
