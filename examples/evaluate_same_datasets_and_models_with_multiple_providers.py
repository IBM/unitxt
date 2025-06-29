import pandas as pd
from unitxt.api import evaluate, load_dataset
from unitxt.artifact import fetch_artifact

df = pd.DataFrame(
    columns=[
        "provider",
        "model",
        "format_as_chat_api",
        "num_instances",
        "score_name",
        "score",
        "ci_low",
        "ci_high",
    ]
)

for provider in [
    "watsonx-sdk",
    "watsonx",
]:
    for model_name in [
        "granite-3-8b-instruct",
        "llama-3-3-70b-instruct",
    ]:
        for format_as_chat_api in [True]:
            if format_as_chat_api:
                format = "formats.chat_api"
            card, _ = fetch_artifact("cards.sst2")

            dataset = load_dataset(
                card=card,
                template_card_index=0,
                format=format,
                num_demos=1,
                demos_pool_size=10,
                loader_limit=1000,
                max_test_instances=10,
                use_cache=True,
                split="test",
            )

            from unitxt.inference import CrossProviderInferenceEngine

            model = CrossProviderInferenceEngine(
                model=model_name, max_tokens=1024, provider=provider
            )
            predictions = model(dataset)

            results = evaluate(predictions=predictions, data=dataset)
            # import pandas as pd
            # result_df = pd.json_normalize(evaluated_dataset)
            # result_df.to_csv(f"output.csv")
            # Print results
            print(results.instance_scores.summary)

            global_scores = results.global_scores
            df.loc[len(df)] = [
                provider,
                model_name,
                format_as_chat_api,
                global_scores["num_of_instances"],
                global_scores["score_name"],
                global_scores["score"],
                global_scores["score_ci_low"],
                global_scores["score_ci_high"],
            ]

            df = df.round(decimals=2)
            print(df.to_markdown())
