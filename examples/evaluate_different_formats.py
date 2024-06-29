from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.inference import IbmGenAiInferenceEngine, IbmGenAiInferenceEngineParams
from unitxt.text_utils import print_dict

logger = get_logger()


model_name = "meta-llama/llama-3-70b-instruct"
gen_params = IbmGenAiInferenceEngineParams(max_new_tokens=32)
inference_model = IbmGenAiInferenceEngine(model_name=model_name, parameters=gen_params)
card = "cards.boolq"
template = "templates.classification.multi_class.title"
for format in [
    "formats.llama3_instruct",
    "formats.empty",
    "formats.llama3_instruct_all_demos_in_one_turn",
]:
    for system_prompt in ["system_prompts.models.llama2", "system_prompts.empty"]:
        dataset = load_dataset(
            card=card,
            template=template,
            format=format,
            system_prompt=system_prompt,
            num_demos=4,
            demos_pool_size=100,
            loader_limit=200,
        )

        test_dataset = dataset["test"]

        predictions = inference_model.infer(test_dataset)
        evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

        logger.info(
            f"Results for format '{format}' and system prompt '{system_prompt}':"
        )
        print_dict(
            evaluated_dataset[0],
            keys_to_print=[
                "source",
                "prediction",
                "score",
            ],
        )
