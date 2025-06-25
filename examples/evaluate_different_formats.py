import json
import time

import pandas as pd
from unitxt.api import evaluate, load_dataset
from unitxt.inference import (
    CrossProviderInferenceEngine,
    WMLInferenceEngineChat,
    WMLInferenceEngineGeneration,
)

print("Creating cross_provider_rits ...")
cross_provider_rits = CrossProviderInferenceEngine(
    model="granite-3-8b-instruct", max_tokens=32, provider="rits", temperature=0
)

print("Creating cross_provider_watsonx ...")
cross_provider_watsonx = CrossProviderInferenceEngine(
    model="granite-3-8b-instruct", max_tokens=32, provider="watsonx", temperature=0
)
print("Creating wml_gen ...")
wml_gen = WMLInferenceEngineGeneration(
    model_name="ibm/granite-3-8b-instruct", max_new_tokens=32, temperature=0
)
print("Creating wml_chat ...")
wml_chat = WMLInferenceEngineChat(
    model_name="ibm/granite-3-8b-instruct", max_tokens=32, temperature=0
)

df = pd.DataFrame(
    columns=[
        "model",
        "format",
        "system_prompt",
        "f1_micro",
        "ci_low",
        "ci_high",
        "duration",
        "num_instances",
        "type_of_input",
    ]
)

model_list = [
    (cross_provider_watsonx, "cross-provider-watsonx"),
    (wml_chat, "wml-chat"),
    (wml_gen, "wml-gen"),
]

# This example compares the impact of different formats on a classification dataset
#
# formats.chat_api  - creates a list of OpenAI messages, where the instruction appears in the system prompt.
#
# [
#    {
#        "role": "system",
#        "content": "Classify the contractual clauses of the following text to one of these options: Records, Warranties... "
#    },
#    {
#        "role": "user",
#        "content": "text: Each Credit Party shall maintain..."
#    },
#    {
#        "role": "assistant",
#        "content": "The contractual clauses is Records"
#    },
#    {
#        "role": "user",
#        "content": "text: Executive agrees to be employed with the Company...."
#    }
# ]
#
# formats.chat_api[place_instruction_in_user_turns=True] - creates a list of OpenAI messages, where the instruction appears in each user turn prompt.
#
# [
#     {
#         "role": "user",
#         "content": "Classify the contractual clauses of the following text to one of these options: ...
#                      text: Each Credit Party shall maintain...."
#     },
#     {
#         "role": "assistant",
#         "content": "The contractual clauses is Records"
#     },
#     {
#         "role": "user",
#         "content": "Classify the contractual clauses of the following text to one of these options: ...
#                     text: Executive agrees to be employed with the Company...
#     }
# ]
#
# formats.empty  - pass inputs as a single string
#
# "Classify the contractual clauses of the following text to one of these options: Records, Warranties,.
# text: Each Credit Party shall maintain...
# The contractual clauses is Records
#
# text: Executive agrees to be employed with the Company,...
# The contractual clauses is "

for model, model_name in model_list:
    print(model_name)
    card = "cards.ledgar"
    template = "templates.classification.multi_class.instruction"
    for format in [
        "formats.chat_api[place_instruction_in_user_turns=True]",
        "formats.chat_api",
        "formats.empty",
    ]:
        for system_prompt in [
            "system_prompts.empty",
        ]:
            if model_name == "wml-gen" and "formats.chat_api" in format:
                continue
            if model_name == "wml-chat" and "formats.chat_api" not in format:
                continue
            dataset = load_dataset(
                card=card,
                format=format,
                system_prompt=system_prompt,
                template=template,
                num_demos=5,
                demos_pool_size=100,
                loader_limit=1000,
                max_test_instances=128,
                split="test",
            )
            type_of_input = type(dataset[0]["source"])

            print("Starting inference...")
            start = time.perf_counter()
            predictions = model(dataset)
            end = time.perf_counter()
            duration = end - start
            print("End of inference...")

            results = evaluate(predictions=predictions, data=dataset)

            print(
                f"Sample input and output for format '{format}' and system prompt '{system_prompt}':"
            )

            print("Example prompt:")

            print(json.dumps(results.instance_scores[0]["source"], indent=4))

            print("Example prediction:")

            print(json.dumps(results.instance_scores[0]["prediction"], indent=4))

            global_scores = results.global_scores
            df.loc[len(df)] = [
                model_name,
                format,
                system_prompt,
                global_scores["score"],
                global_scores["score_ci_low"],
                global_scores["score_ci_high"],
                duration,
                len(predictions),
                type_of_input,
            ]

            df = df.round(decimals=2)
            print(df.to_markdown())
