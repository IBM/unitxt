from unitxt import evaluate, load_dataset
from unitxt.inference import CrossProviderInferenceEngine
from unitxt.text_utils import print_dict

"""
We are evaluating only on a small subset (by using `max_test_instances=4`), in order for the example to finish quickly.
The dataset full size if around 40k examples. You should use around 1k-4k in your evaluations.
"""
dataset = load_dataset(
    card="cards.arena_hard.generation.english_gpt_4_0314_reference",
    template="template.generation.empty",
    format="formats.chat_api",
    metrics=[
        "metrics.llm_as_judge.pairwise_comparative_rating.llama_3_8b_instruct.template_arena_hard"
    ],
    max_test_instances=4,
    split="test",
)

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
scores = evaluate(predictions=predictions, data=dataset)

print_dict(scores[0]["score"]["global"])
