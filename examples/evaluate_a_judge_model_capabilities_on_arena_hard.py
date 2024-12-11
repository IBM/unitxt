from unitxt import evaluate, load_dataset
from unitxt.inference import CrossProviderInferenceEngine
from unitxt.text_utils import print_dict

"""
We are evaluating only on a small subset (by using `max_test_instances=4`), in order for the example to finish quickly.
The dataset full size if around 40k examples. You should use around 1k-4k in your evaluations.
"""
dataset = load_dataset(
    card="cards.arena_hard.response_assessment.pairwise_comparative_rating.both_games_gpt_4_judge",
    template="templates.response_assessment.pairwise_comparative_rating.arena_hard_with_shuffling",
    format="formats.chat_api",
    max_test_instances=None,
    split="test",
).select(range(5))

inference_model = CrossProviderInferenceEngine(
    model="llama-3-2-1b-instruct", provider="watsonx"
)
"""
We are using a CrossProviderInferenceEngine inference engine that supply api access to provider such as:
watsonx, bam, openai, azure, aws and more.

For the arguments these inference engines can receive, please refer to the classes documentation or read
about the the open ai api arguments the CrossProviderInferenceEngine follows.
"""

predictions = inference_model.infer(dataset)
results = evaluate(predictions=predictions, data=dataset)

print_dict(results.global_scores)
