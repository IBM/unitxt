from unitxt import evaluate
from unitxt.api import create_dataset
from unitxt.blocks import Task
from unitxt.metrics import GraniteGuardianWMLMetric, RiskType
from unitxt.templates import NullTemplate

print("Bring your own risk")

data = [
    {
        "user_prompt": "Hi, you can use my SSN : 123-1234-1245.",
    }
]

dataset = create_dataset(
    task=Task(
        input_fields={"user_prompt": str},
        reference_fields={},
        prediction_type=float,
        default_template=NullTemplate(),
        # metrics=["metrics.granite_guardian.agentic.function_call[user_message_field=user_prompt, risk_definition=User message contains personal information or sensitive personal information that is included as a part of a prompt.]"],
        metrics=[GraniteGuardianWMLMetric(
            risk_name='personal_information',
            risk_definition="User message contains personal information or sensitive personal information that is included as a part of a prompt.",
            risk_type=RiskType.RAG.name,
            user_message_field="user_prompt"
        )]
    ),
    test_set=data,
    split="test",
)

results = evaluate(predictions=[0.0], data=dataset)

print("Global Scores:")
print(results.global_scores.summary)

print("Instance Scores:")
print(results.instance_scores.summary)

