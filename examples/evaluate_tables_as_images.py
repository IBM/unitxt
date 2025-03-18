from unitxt.api import evaluate, load_dataset
from unitxt.inference import CrossProviderInferenceEngine
from unitxt.struct_data_operators import SerializeTableAsImage

# Use the Unitxt APIs to load the wnli entailment dataset using the standard template in the catalog for relation task with 2-shot in-context learning.
# We set loader_limit to 20 to limit reduce inference time.
dataset = load_dataset(
    card="cards.wikitq",
    format="formats.chat_api",
    system_prompt="system_prompts.general.be_concise",
    loader_limit=20,
    serializer=[
        SerializeTableAsImage(),
    ],
    split="test",
)

model = CrossProviderInferenceEngine(model="llama-3-2-11b-vision-instruct")
"""
We are using a CrossProviderInferenceEngine inference engine that supply api access to providers such as:
watsonx, bam, openai, azure, aws and more.
For more information, visit the :ref:`inference engines guide <inference>`
"""
predictions = model(dataset)

results = evaluate(predictions=predictions, data=dataset)

print("Global Results:")
print(results.global_scores.summary)

# print("Instance Results:")
# print(results.instance_scores.summary)
