from unitxt.api import create_dataset, evaluate
from unitxt.inference import CrossProviderInferenceEngine
from unitxt.serializers import ListSerializer
from unitxt.templates import MultiReferenceTemplate
from unitxt.text_utils import print_dict

# Assume the RAG data is proved in this format
data = [
    {
        "question": "What city is the largest in Texas?",
        "contexts": [
            "Austin is the capital of Texas.",
            "Houston is the the largest city in Texas but not the capital of it. ",
        ],
        "reference_answers": ["Houston"],
    },
    {
        "question": "What city is the capital of Texas?",
        "contexts": [
            "Houston is the the largest city in Texas but not the capital of it. "
        ],
        "reference_answers": ["Austin"],
    },
]

template = MultiReferenceTemplate(
    instruction="Answer the question based on the information provided in the document given below.\n\n",
    input_format="Contexts:\n\n{contexts}\n\nQuestion: {question}",
    references_field="reference_answers",
    serializer={"contexts": ListSerializer(separator="\n\n")},
)

# Verbalize the dataset using the template
dataset = create_dataset(
    test_set=data,
    template=template,
    task="tasks.rag.response_generation",
    format="formats.chat_api",
    split="test",
)

engine = CrossProviderInferenceEngine(model="llama-3-2-1b-instruct", provider="watsonx")
# The provider can be one of: ["watsonx", "together-ai", "open-ai", "aws", "ollama", "bam"]


predictions = engine.infer(dataset)
evaluated_dataset = evaluate(predictions=predictions, data=dataset)

# Print results
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
