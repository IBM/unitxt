from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.inference import IbmGenAiInferenceEngine, IbmGenAiInferenceEngineParams
from unitxt.text_utils import print_dict

logger = get_logger()


model_name = "meta-llama/llama-3-8b-instruct"
gen_params = IbmGenAiInferenceEngineParams(max_new_tokens=32)
inference_model = IbmGenAiInferenceEngine(model_name=model_name, parameters=gen_params)
card = "cards.boolq.classification"
template = "templates.classification.multi_class.relation.default"

all_scores = {}
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
            num_demos=2,
            demos_pool_size=100,
            loader_limit=1000,
            max_test_instances=300,
        )

        test_dataset = dataset["test"]

        predictions = inference_model.infer(test_dataset)
        evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

        logger.info(
            f"Sample input and output for format '{format}' and system prompt '{system_prompt}':"
        )
        print_dict(
            evaluated_dataset[0],
            keys_to_print=[
                "source",
                "prediction",
            ],
        )
        global_scores = evaluated_dataset[0]["score"]["global"]
        print_dict(
            global_scores,
            keys_to_print=["score_name", "score", "score_ci_low", "score_ci_high"],
        )
        all_scores[(format, system_prompt)] = global_scores


for (format, system_prompt), global_scores in all_scores.items():
    logger.info(f"**** score for format '{format}' and system prompt '{system_prompt}'")
    logger.info(
        f"**** {global_scores['score_name']} : {global_scores['score']} - 95% confidence internal [{global_scores['score_ci_low']},{global_scores['score_ci_high']}]"
    )
