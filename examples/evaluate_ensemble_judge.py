from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.blocks import Task, TaskCard
from unitxt.loaders import LoadFromDictionary
from unitxt.templates import InputOutputTemplate, TemplatesDict

logger = get_logger()

data = {
    "test": [
        {"conversation": "user: is eugene from walking dead really a scientist"},
        {"conversation": "user: is eugene from walking dead really a scientist"},
        {
            "conversation": "user: is eugene from walking dead really a scientist\nagent: No, Eugene from The Walking Dead is not a real scientist. He lied about being a scientist to manipulate the other survivors into taking him to Washington D.C., believing it to be the best chance for survival. In reality, he is a high school science teacher who does not know how to cure the virus. However, he is shown to be highly intelligent and resourceful, able to do things such as repairing the solar power system in Alexandria and making bullets for the Saviors.\nuser: Why did he lied about himself being a scientist?"
        },
    ]
}

predictions = [
    "In the Walking Dead series, Eugene Porter initially claims to be a scientist who knows the cure to the zombie plague. However, it is eventually revealed that he is not a scientist, but a high school science teacher who lied about knowing the cure to manipulate other survivors into helping him.",
    "thank you for the question.",
    "Eugene lied about being a scientist to gain the trust and protection of Abraham and Rosita, as well as to believe that going to Washington D.C. was the best chance for survival. He was not a scientist and did not know how to cure the virus, but he used his intelligence and resourcefulness to help the group survive in other ways.",
]

card = TaskCard(
    loader=LoadFromDictionary(data=data),
    task=Task(
        input_fields={"conversation": "str"},
        reference_fields={},
        prediction_type="str",
        metrics=[
            "metrics.llm_as_judge.conversation_answer_topicality.ensemble_v1_ibmgenai_judges"
        ],
    ),
    templates=TemplatesDict(
        {
            "simple": InputOutputTemplate(
                input_format="{conversation}",
                output_format="",
            )
        }
    ),
)

test_dataset = load_dataset(card=card, template_card_index="simple")["test"]
results = evaluate(predictions=predictions, data=test_dataset)

print("Global Results:")
print(results.global_scores.summary)

print("Instance Results:")
print(results.instance_scores.summary)
