from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.blocks import Task, TaskCard
from unitxt.loaders import LoadFromDictionary
from unitxt.templates import InputOutputTemplate, TemplatesDict
from unitxt.text_utils import print_dict

logger = get_logger()


data = {
    "test": [
        {
            "inquiry": "Where can I find more information on health centers?",
        },
        {
            "inquiry": "How do I connect to customer care representative?",
        },
        {
            "inquiry": "How do I connect to customer care representative?",
        },
    ]
}

predictions = [
    "I am sorry, but the provided document does not contain answer to your question.",
    "The document does not provide a specific answer to your question.",
    "Hello, you can chat with a representative by clicking on the chat icon at the top of the page.",
]

card = TaskCard(
    loader=LoadFromDictionary(data=data),
    task=Task(
        input_fields={"inquiry": "str"},
        reference_fields={},
        prediction_type="str",
        metrics=[
            "metrics.llm_as_judge.conversation_answer_idk.llama3_v1_ibmgenai_judges"
        ],
    ),
    templates=TemplatesDict(
        {
            "simple": InputOutputTemplate(
                input_format="{inquiry}",
                output_format="",
            )
        }
    ),
)

test_dataset = load_dataset(card=card, template_card_index="simple")["test"]
evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

for instance in evaluated_dataset:
    print_dict(
        instance,
        keys_to_print=[
            "source",
            "prediction",
            "processed_prediction",
            "references",
            "score",
        ],
    )
