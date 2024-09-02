from unitxt.api import evaluate, load_dataset
from unitxt.inference_engine import HFPipelineBasedInferenceEngine
from unitxt.text_utils import print_dict

# Use the Unitxt APIs to load the wnli entailment dataset using the standard template in the catalog for relation task with 2-shot in-context learning.
# We set loader_limit to 20 to limit reduce inference time.
dataset = load_dataset(
    card="cards.wnli",
    template="templates.classification.multi_class.relation.default",
    num_demos=2,
    demos_pool_size=10,
    loader_limit=20,
)

test_dataset = dataset["test"]

# Infer using flan t5 base using HF API, can be replaced with any
# inference code.
#
# change to this to infer with IbmGenAI APIs:
#
# from unitxt.inference import IbmGenAiInferenceEngine
# inference_model = IbmGenAiInferenceEngine(model_name=model_name, max_new_tokens=32)
#
# or this to infer using WML APIs:
#
# from unitxt.inference import WMLInferenceEngine
# inference_model = WMLInferenceEngine(model_name=model_name, max_new_tokens=32)
#
# or to this to infer using OpenAI APIs:
#
# from unitxt.inference import OpenAiInferenceEngine
# inference_model = OpenAiInferenceEngine(model_name=model_name, max_new_tokens=32)
#
# Note that to run with OpenAI APIs you need to change the loader specification, to
# define that your data can be sent to a public API:
#
# loader=LoadFromDictionary(data=data,data_classification_policy=["public"]),

model_name = "google/flan-t5-base"
inference_model = HFPipelineBasedInferenceEngine(
    model_name=model_name, max_new_tokens=32
)
predictions = inference_model.infer(test_dataset)

evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

# Print results
print_dict(
    evaluated_dataset[0],
    keys_to_print=[
        "source",
        "prediction",
        "processed_prediction",
        "references",
        "score",
    ],
)
