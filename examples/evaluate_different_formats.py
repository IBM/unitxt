import pandas as pd
from unitxt.api import evaluate, load_dataset
from unitxt.inference import CrossProviderInferenceEngine

model = CrossProviderInferenceEngine(
    model="llama-3-8b-instruct", max_tokens=32, provider="bam"
)
"""
We are using a CrossProviderInferenceEngine inference engine that supply api access to provider such as:
watsonx, bam, openai, azure, aws and more.

For the arguments these inference engines can receive, please refer to the classes documentation or read
about the the open ai api arguments the CrossProviderInferenceEngine follows.
"""

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
            split="test",
        )

        predictions = model(dataset)
        results = evaluate(predictions=predictions, data=dataset)

        print(
            f"Sample input and output for format '{format}' and system prompt '{system_prompt}':"
        )

        print(
            results.instance_scores.to_df(
                columns=[
                    "source",
                    "prediction",
                ]
            )
        )

        global_scores = results.global_scores
        df.loc[len(df)] = [
            format,
            system_prompt,
            global_scores["score"],
            global_scores["score_ci_low"],
            global_scores["score_ci_high"],
        ]

        df = df.round(decimals=2)
        print(df.to_markdown())
