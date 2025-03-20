import json

from unitxt import get_logger, load_dataset
from unitxt.api import LoadFromDictionary, TaskCard, evaluate
from unitxt.blocks import Rename
from unitxt.inference import HFPipelineBasedInferenceEngine
from unitxt.operators import IndexOf, ListFieldValues
from unitxt.templates import MultipleChoiceTemplate

logger = get_logger()

# Set up question answer pairs in a dictionary
data = [
    {"Question": "What is the capital of Texas?", "Option A": "Austin", "Option B": "Houston", "Option C": "Dallas", "Answer" : "Austin"},
    {"Question": "What is the color of the sky?", "Option A": "Pink",   "Option B":  "Red", "Option C": "Blue" , "Answer" : "Blue"},
]


# Create a unitxt cards that converts the input data to the format required by the
# t`asks.qa.multiple_choice.open task`.
#
# It concatenates the different options fields to the 'choices' field.
# And sets the 'answer' field, to the index of the correct answer in the 'choices' field.
card =  TaskCard(
        loader=LoadFromDictionary(data = { "test": data }),
        preprocess_steps=[
            Rename(
                field_to_field={"Answer": "answer", "Question" : "question"},
            ),
            ListFieldValues(fields=["Option A", "Option B", "Option C"], to_field="choices"),
            IndexOf(search_in="choices", index_of="answer", to_field="answer")
        ],
        task="tasks.qa.multiple_choice.open"
)

template = MultipleChoiceTemplate(
        input_format="Answer the following question, returning only a single letter.  Do not any add any explanations. \n\nQuestion: {question}\nAnswers:\n{choices}\nAnswer:",
        target_field="answer",
        choices_separator="\n",
        postprocessors=["processors.lower_case","processors.first_character"],
    )

dataset = load_dataset(
    card = card,
    template=template,
    split="test",
    format="formats.chat_api",
)

# Infer using Llama-3.2-1B base using HF API
model = HFPipelineBasedInferenceEngine(
    model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct", max_new_tokens=32
)
# Change to this to infer with external APIs:
#from unitxt.inference import CrossProviderInferenceEngine
# model = CrossProviderInferenceEngine(model="llama-3-2-1b-instruct", provider="watsonx")
# The provider can be one of: ["watsonx", "together-ai", "open-ai", "aws", "ollama", "bam"]


predictions = model(dataset)
results = evaluate(predictions=predictions, data=dataset)

print("Example prompt:")
print(json.dumps(results.instance_scores[0]["source"], indent=4))


print("Instance Results:")
print(results.instance_scores)

print("Global Results:")
print(results.global_scores.summary)


