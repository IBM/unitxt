from unitxt.api import evaluate, load_dataset
from unitxt.inference import CrossProviderInferenceEngine

# Use the Unitxt APIs to load the wnli entailment dataset using the standard template in the catalog for relation task with 2-shot in-context learning.
# We set loader_limit to 20 to limit reduce inference time.
dataset = load_dataset(
    card="cards.wnli",
    template="templates.classification.multi_class.relation.default",
    format="formats.chat_api",
    num_demos=2,
    demos_pool_size=10,
    loader_limit=20,
    split="test",
)

model = CrossProviderInferenceEngine(model="llama-3-2-1b-instruct", provider="watsonx")
"""
We are using a CrossProviderInferenceEngine inference engine that supply api access to providers such as:
watsonx, bam, openai, azure, aws and more.
For more information, visit the :ref:`inference engines guide <inference>`
"""
predictions = model(dataset)

results = evaluate(predictions=predictions, data=dataset)

print(
    results.instance_scores.to_df(
        columns=[
            "source",
            "prediction",
            "processed_prediction",
            "references",
            "score",
        ]
    )
)
