import pandas as pd
from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.inference_engines import IbmGenAiInferenceEngine
from unitxt.text_utils import print_dict

logger = get_logger()


model_name = "meta-llama/llama-3-8b-instruct"
inference_model = IbmGenAiInferenceEngine(model_name=model_name, max_new_tokens=32)
card = "cards.boolq.classification"
template = "templates.classification.multi_class.relation.default"

df = pd.DataFrame(columns=["format", "system_prompt", "f1_micro", "ci_low", "ci_high"])

for format in [
    "formats.llama3_instruct",
    "formats.empty",
    "formats.llama3_instruct_all_demos_in_one_turn",
]:
    for system_prompt in [
        "system_prompts.models.llama2",
        "system_prompts.empty",
    ]:
        dataset = load_dataset(
            card=card,
            template=template,
            format=format,
            system_prompt=system_prompt,
            num_demos=2,
            demos_pool_size=50,
            loader_limit=300,
            max_test_instances=100,
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
        df.loc[len(df)] = [
            format,
            system_prompt,
            global_scores["score"],
            global_scores["score_ci_low"],
            global_scores["score_ci_high"],
        ]

        df = df.round(decimals=2)
        logger.info(df.to_markdown())
