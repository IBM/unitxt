import pandas as pd
from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.inference import IbmGenAiInferenceEngine, IbmGenAiInferenceEngineParams
from unitxt.text_utils import print_dict

logger = get_logger()

# Run inference on mnli (entailment task) on the two templates with both 0 and 3 shot in context learning.
card = "cards.sst2"
model_name = "meta-llama/llama-3-8b-instruct"
model_name = "google/flan-t5-xxl"
gen_params = IbmGenAiInferenceEngineParams(max_new_tokens=32)
inference_model = IbmGenAiInferenceEngine(model_name=model_name, parameters=gen_params)
df = pd.DataFrame(columns=["template", "augmentation", "f1_micro", "ci_low", "ci_high"])


for template in [
    "templates.classification.multi_class.default",
]:
    for augmentor in [
        #        "augmentors.no_augmentation",
        #        "augmentors.augment_whitespace_prefix_and_suffix_task_input",
        "augmentors.augment_whitespace_model_input"
        # "augmentors.augmentors.my_augmentor"
    ]:
        dataset = load_dataset(
            card=card,
            template=template,
            num_demos=3,
            demos_pool_size=100,
            loader_limit=500,
            max_test_instances=300,
            format="formats.llama3_instruct",
            augmentor=augmentor,
        )

        test_dataset = dataset["test"]
        predictions = inference_model.infer(test_dataset)
        evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

        logger.info(
            f"Sample input and output for template '{template}' and augmentor '{augmentor}':"
        )
        print_dict(
            evaluated_dataset[0],
            keys_to_print=["source", "prediction", "processed_prediction"],
        )
        global_scores = evaluated_dataset[0]["score"]["global"]
        print_dict(
            global_scores,
            keys_to_print=["score_name", "score", "score_ci_low", "score_ci_high"],
        )
        df.loc[len(df)] = [
            template,
            augmentor,
            global_scores["score"],
            global_scores["score_ci_low"],
            global_scores["score_ci_high"],
        ]

df = df.round(decimals=2)
logger.info(df.to_markdown())
