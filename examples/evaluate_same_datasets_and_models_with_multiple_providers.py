import pandas as pd
from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.artifact import fetch_artifact
from unitxt.formats import SystemFormat
from unitxt.text_utils import print_dict

logger = get_logger()

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
        "llama-3-8b-instruct",
    ]:
        for format_as_chat_api in [True, False]:
            if format_as_chat_api and provider == "watsonx-sdk":
                continue
            if format_as_chat_api:
                format = "formats.chat_api"
            else:
                if model_name.startswith("llama"):
                    format = "formats.llama3_instruct"
                if model_name.startswith("granite"):
                    format = SystemFormat(
                        demo_format=(
                            "{instruction}\\N{source}\\N<|end_of_text|>\n"
                            "<|start_of_role|>assistant<|end_of_role|>{target}\\N<|end_of_text|>\n"
                            "<|start_of_role|>user<|end_of_role|>"
                        ),
                        model_input_format=(
                            "<|start_of_role|>system<|end_of_role|>{system_prompt}<|end_of_text|>\n"
                            "<|start_of_role|>user<|end_of_role|>{demos}{instruction}\\N{source}\\N<|end_of_text|>\n"
                            "<|start_of_role|>assistant<|end_of_role|>"
                        ),
                    )
            card, _ = fetch_artifact("cards.sst2")

            dataset = load_dataset(
                card=card,
                template_card_index=0,
                format=format,
                num_demos=1,
                demos_pool_size=10,
                loader_limit=1000,
                max_test_instances=10,
                disable_cache=False,
                split="test",
            )

            from unitxt.inference import CrossProviderInferenceEngine

            inference_model = CrossProviderInferenceEngine(
                model=model_name, max_tokens=1024, provider=provider
            )
            predictions = inference_model.infer(dataset)

            evaluated_dataset = evaluate(predictions=predictions, data=dataset)
            # import pandas as pd
            # result_df = pd.json_normalize(evaluated_dataset)
            # result_df.to_csv(f"output.csv")
            # Print results
            print_dict(
                evaluated_dataset[0],
                keys_to_print=[
                    "source",
                    "prediction",
                    "processed_prediction",
                    "processed_references",
                ],
            )

            global_scores = evaluated_dataset[0]["score"]["global"]
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
            logger.info(df.to_markdown())
