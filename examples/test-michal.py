from datasets import load_dataset
from unitxt import get_logger, get_settings
from unitxt.api import evaluate

from datasets import disable_caching
from unitxt.inference import IbmGenAiInferenceEngineParams, IbmGenAiInferenceEngine
from unitxt import evaluate, dataset_utils
from unitxt.standard import StandardRecipe

from unitxt.text_utils import print_dict

logger = get_logger()
settings = get_settings()
settings.allow_unverified_code = True

card = "card=cards.sap_summarization,template_card_index=1,loader_limit=20" #metrics=[metrics.llm_as_judge.rating.llama_3_70b_instruct_ibm_genai_template_generic_single_turn]


dataset_params = dataset_utils.parse(card)
disable_caching()
recipe = StandardRecipe(**dataset_params)
stream = recipe()
test_dataset = stream.to_dataset()['test'] # .shuffle(seed=42).select(range(4))

params = IbmGenAiInferenceEngineParams(max_new_tokens=1024, random_seed=42)
model = "meta-llama/llama-3-70b-instruct"
inference_model = IbmGenAiInferenceEngine(model_name=model, parameters=params)

predictions = inference_model.infer(test_dataset)

# Evaluate the predictions using the defined metric.
evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

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