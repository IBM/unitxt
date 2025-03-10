import json
import time

import pandas as pd
from lh_eval_api import LakeHouseLoader
from unitxt.api import evaluate, load_dataset
from unitxt.inference import (
    CrossProviderInferenceEngine,
    WMLInferenceEngineChat,
    WMLInferenceEngineGeneration,
)

x = LakeHouseLoader # To avoid warnings, of unused imports.
print("Creating cross_provider_rits ...")
cross_provider_rits = CrossProviderInferenceEngine(model="granite-3-8b-instruct", max_tokens=32, provider="rits")

print("Creating cross_provider_watsonx ...")
cross_provider_watsonx = CrossProviderInferenceEngine(model="granite-3-8b-instruct", max_tokens=32, provider="watsonx")
print("Creating wml_gen ...")
wml_gen= WMLInferenceEngineGeneration(model_name="ibm/granite-3-8b-instruct",max_new_tokens=32)
print("Creating wml_chat ...")
wml_chat=   WMLInferenceEngineChat(model_name="ibm/granite-3-8b-instruct",max_tokens=32,top_logprobs=None)

#wml_chat = WMLInferenceEngineChat(
#    model_name="ibm/granite-vision-3-2-2b",max_tokens=32,top_logprobs=None
#)
#wml_gen= WMLInferenceEngineGeneration(model_name="ibm/granite-vision-3-2-2b",max_new_tokens=32)

df = pd.DataFrame(columns=["model","format", "system_prompt", "f1_micro", "ci_low", "ci_high", "duration", "num_instances","type_of_input"])

#model_list = [(cross_provider_rits,"cross_provider_rits"),(wml_chat,"wml-chat"),(cross_provider_watsonx, "cross-provider-watsonx")]
#model_list = [(cross_provider_watsonx, "cross-provider-watsonx"),(cross_provider_rits,"cross_provider_rits")]
model_list = [(cross_provider_watsonx, "cross-provider-watsonx"),(wml_chat,"wml-chat"),(wml_gen,"wml-gen") ]
#model_list = [(cross_provider_rits,"cross_provider_rits")]
for (model,model_name) in model_list:
    print(model_name)
    card = "cards.cat"
    template = "templates.classification.multi_label.instruct_question_select"

    for format in [
        "formats.chat_api[repeat_instruction_per_turn=True,add_target_prefix=False]",
        "formats.chat_api[repeat_instruction_per_turn=True]",
        "formats.granite_instruct_custom",
        "formats.chat_api",
#        "formats.empty",
    ]:
        for system_prompt in [
            "system_prompts.models.granite_instruct_classify",
#            "system_prompts.empty",
        ]:
            if (model_name == "wml-gen" and  "formats.chat_api" in format):
                continue
            if (model_name == "wml-chat" and  "formats.chat_api" not in format):
                continue
            dataset = load_dataset(
                card=card,
                template=template,
                format=format,
                system_prompt=system_prompt,
                num_demos=5,
                demos_pool_size=100,
                loader_limit=1000,
                max_test_instances=128  ,
                split="test",
            )
            type_of_input = (type(dataset[0]["source"]))

            print("Starting inference...")
            start = time.perf_counter()
            predictions = model(dataset)
            end = time.perf_counter()
            duration = end-start
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
                type_of_input
            ]

            df = df.round(decimals=2)
            print(df.to_markdown())
