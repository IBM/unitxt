from unitxt import evaluate
from unitxt.inference import (
    IbmGenAiInferenceEngine,
    IbmGenAiInferenceEngineParams,
)
from unitxt.standard import StandardRecipe

model_id = "meta-llama/llama-3-70b-instruct"
model_format = "formats.llama3_chat"

dataset = (
    StandardRecipe(
        card="cards.arena_hard.response_assessment.pairwise_comparative_rating.both_games_gpt_4_judge",
        template="templates.response_assessment.pairwise_comparative_rating.arena_hard_with_shuffling",
        format=model_format,
    )()
    .to_dataset()["test"]
    .shuffle()
    .select(range(4))
)
# We are evaluating only on a small subset, in order for the example to finish quickly.
# The dataset full size if around 40k examples. You should use around 1k-4k in your evaluations.

params = IbmGenAiInferenceEngineParams(max_new_tokens=1024, random_seed=42)
inference_model = IbmGenAiInferenceEngine(model_name=model_id, parameters=params)

# Using OpenAi model
# params = OpenAiInferenceEngineParams(max_new_tokens=1024)
# inference_model = OpenAiInferenceEngine(model_name=model, parameters=params)

# Using Huggingface model
# inference_model = HFPipelineBasedInferenceEngine(model_name=model_id, max_new_tokens=1024)

predictions = inference_model.infer(dataset)
scores = evaluate(predictions=predictions, data=dataset)
# [print(item) for item in scores[0]["score"]["global"].items()]
