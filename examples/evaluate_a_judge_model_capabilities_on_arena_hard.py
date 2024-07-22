from unitxt import evaluate, load_dataset
from unitxt.inference import MockInferenceEngine
from unitxt.text_utils import print_dict

model_id = "meta-llama/llama-3-70b-instruct"
model_format = "formats.llama3_instruct"

"""
We are evaluating only on a small subset (by using "select(range(4)), in order for the example to finish quickly.
The dataset full size if around 40k examples. You should use around 1k-4k in your evaluations.
"""
dataset = load_dataset(
    card="cards.arena_hard.response_assessment.pairwise_comparative_rating.both_games_gpt_4_judge",
    template="templates.response_assessment.pairwise_comparative_rating.arena_hard_with_shuffling",
    format=model_format,
)["test"].select(range(4))

inference_model = MockInferenceEngine(model_name=model_id)
"""
We are using a mock inference engine (and model) in order for the example to finish quickly.
In real scenarios you can use model from Huggingface, OpenAi, and IBM, using the following:
from unitxt.inference import (HFPipelineBasedInferenceEngine, IbmGenAiInferenceEngine, OpenAiInferenceEngine)
and switch them with the MockInferenceEngine class in the example.
For the arguments these inference engines can receive, please refer to the classes documentation.

Example of using an IBM model:
from unitxt.inference import (IbmGenAiInferenceEngine, IbmGenAiInferenceEngineParamsMixin)
params = IbmGenAiInferenceEngineParamsMixin(max_new_tokens=1024, random_seed=42)
inference_model = IbmGenAiInferenceEngine(model_name=model_id, parameters=params)
"""

predictions = inference_model.infer(dataset)
scores = evaluate(predictions=predictions, data=dataset)

print_dict(scores[0]["score"]["global"])
