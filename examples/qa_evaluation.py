from unitxt import get_logger
from unitxt.api import create_dataset, evaluate
from unitxt.inference import CrossProviderInferenceEngine

logger = get_logger()

# Set up question answer pairs in a dictionary
test_set = [
    {"question": "What is the capital of Texas?", "answers": ["Austin"]},
    {"question": "What is the color of the sky?", "answers": ["Blue"]},
]


# Verbalize the dataset using the catalog template which adds an instruction "Answer the question.",
# and "Question:"/"Answer:" prefixes.
#
# "Answer the question.
#  Question:
#  What is the color of the sky?
#  Answer:
# "

dataset = create_dataset(
    task="tasks.qa.open",
    test_set=test_set,
    template="templates.qa.open",
    split="test",
    format="formats.chat_api",
)

model = CrossProviderInferenceEngine(model="SmolLM2-1.7B-Instruct", provider="hf")
# The provider can be one of: ["watsonx", "together-ai", "open-ai", "aws", "ollama", "hf"]
# (model must be available in the provider service)


predictions = model(dataset)
results = evaluate(predictions=predictions, data=dataset)

print("Global Results:")
print(results.global_scores.summary)

print("Instance Results:")
print(results.instance_scores.summary)
