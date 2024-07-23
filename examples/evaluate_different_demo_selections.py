import pandas as pd
from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.inference import IbmGenAiInferenceEngine
from unitxt.splitters import CloseTextSampler, FixedIndicesSampler, RandomSampler
from unitxt.text_utils import print_dict

logger = get_logger()

# This examples evaluates different kinds of demo selection strategies on a classification task.
# The different strategies are evaluates in 1,3,5 shots. The examples are selected from a demo pool of 100 examples.
# RandomSampler - randomly sample a different set of examples for each test instance
# CloseTextSampler - select the lexically closest amples from the demo pool for each test instance
# FixedIndicesSampler - selec the same fixed set of demo examples for all instances

card = "cards.ledgar"
model_name = "google/flan-t5-xxl"
inference_model = IbmGenAiInferenceEngine(model_name=model_name, max_new_tokens=32)


df = pd.DataFrame(columns=["num_demos", "sampler", "f1_micro", "ci_low", "ci_high"])

for num_demos in [1, 3, 5]:
    for demo_sampler in [
        RandomSampler(),
        CloseTextSampler(field="text"),
        FixedIndicesSampler(indices=[0, 1, 2, 4, 5]),
    ]:
        dataset = load_dataset(
            card=card,
            template="templates.classification.multi_class.title",
            num_demos=num_demos,
            demos_pool_size=300,
            loader_limit=400,
            max_test_instances=200,
            sampler=demo_sampler,
        )

        test_dataset = dataset["test"]

        predictions = inference_model.infer(test_dataset)
        evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

        logger.info(
            f"Sample input and output for sampler {demo_sampler} and num_demos '{num_demos}':"
        )
        print_dict(
            evaluated_dataset[0],
            keys_to_print=["source", "prediction", "processed_prediction"],
        )
        global_scores = evaluated_dataset[0]["score"]["global"]

        df.loc[len(df)] = [
            num_demos,
            demo_sampler.to_json(),
            global_scores["score"],
            global_scores["score_ci_low"],
            global_scores["score_ci_high"],
        ]

        df = df.round(decimals=2)
        logger.info(df.to_markdown())
