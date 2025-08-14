# this python script shows an example of running speech translation evaluation for Granite Speech

from unitxt import evaluate, load_dataset
from unitxt.inference import (
    HFGraniteSpeechInferenceEngine,
)
from unitxt.system_prompts import TextualSystemPrompt

debug = True  # True for extra printing, set to False when commenting out max_test_instances below
max_test_instances = 8

# the available calanguages for the covost2 dataset dataset, are:
#  translation from English to target language:
#   de        German
#   ja        Japanese
#  translation from source language to English:
#   de        German
#   es        Spanish
#   fr        French
#   pt        Portuguese
test_dataset = load_dataset(  # select (un-comment) one of the test sets below
    card="cards.covost2.from_en.en_de",
    # card="cards.covost2.from_en.en_ja",
    # card="cards.covost2.to_en.de_en",
    # card="cards.covost2.to_en.es_en",
    # card="cards.covost2.to_en.fr_en",
    # card="cards.covost2.to_en.pt_en",
    split="test",
    format="formats.chat_api",
    max_test_instances=max_test_instances,  # comment out for running the entire test
    system_prompt=TextualSystemPrompt(
        text="Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
    ),
)

if debug:
    print(">>>>>>>>>>>>>>  first test references  >>>>>>>>>>>>")
    for idx in range(max_test_instances):
        print(f">>>>>>   references {idx}:  ", test_dataset["references"][idx])

model = HFGraniteSpeechInferenceEngine(
    model_name="ibm-granite/granite-speech-3.3-8b",  # two options for Granite Speech 3.3:  2b  and  8b
    max_new_tokens=120,  # 200 for 2b,  120 for 8b
)

predictions = model(test_dataset)

if debug:  # print translation reference texts for debug and inspection
    print(">>>>>>>>>>>>>>  first predictions  >>>>>>>>>>>>")
    for idx in range(max_test_instances):
        print(f">>>>>>>>>>> {idx}:   ", predictions[idx])

results = evaluate(
    predictions=predictions, data=test_dataset, calc_confidence_intervals=False
)

print("Global scores:")
print(results.global_scores.summary)
