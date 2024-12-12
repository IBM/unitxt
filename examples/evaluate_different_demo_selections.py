import pandas as pd
from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.inference import CrossProviderInferenceEngine
from unitxt.splitters import CloseTextSampler, FixedIndicesSampler, RandomSampler

logger = get_logger()

# This examples evaluates different kinds of demo selection strategies on a classification task.
# The different strategies are evaluates in 1,3,5 shots. The examples are selected from a demo pool of 100 examples.
# RandomSampler - randomly sample a different set of examples for each test instance
# CloseTextSampler - select the lexically closest amples from the demo pool for each test instance
# FixedIndicesSampler - selec the same fixed set of demo examples for all instances

model = CrossProviderInferenceEngine(model="llama-3-2-1b-instruct", max_tokens=32)
"""
We are using a CrossProviderInferenceEngine inference engine that supply api access to provider such as:
watsonx, bam, openai, azure, aws and more.

For the arguments these inference engines can receive, please refer to the classes documentation or read
about the the open ai api arguments the CrossProviderInferenceEngine follows.
"""

df = pd.DataFrame(columns=["num_demos", "sampler", "f1_micro", "ci_low", "ci_high"])

for num_demos in [1, 2]:
    for demo_sampler in [
        RandomSampler(),
        CloseTextSampler(field="text"),
        FixedIndicesSampler(indices=[0, 1]),
    ]:
        dataset = load_dataset(
            card="cards.ledgar",
            template="templates.classification.multi_class.title",
            format="formats.chat_api",
            num_demos=num_demos,
            demos_pool_size=50,
            loader_limit=200,
            max_test_instances=10,
            sampler=demo_sampler,
            split="test",
        )

        predictions = model.infer(dataset)
        results = evaluate(predictions=predictions, data=dataset)

        logger.info(
            f"Sample input and output for sampler {demo_sampler} and num_demos '{num_demos}':"
        )
        print(
            results.instance_scores.to_df(
                columns=["source", "prediction", "processed_prediction"]
            )
        )

        global_scores = results.global_scores

        df.loc[len(df)] = [
            num_demos,
            demo_sampler.to_json(),
            global_scores["score"],
            global_scores["score_ci_low"],
            global_scores["score_ci_high"],
        ]

        df = df.round(decimals=2)
        logger.info(df.to_markdown())
