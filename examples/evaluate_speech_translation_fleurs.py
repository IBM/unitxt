# this python script shows an example of running speech translation evaluation for Granite Speech

from unitxt import evaluate, load_dataset
from unitxt.inference import (
    HFGraniteSpeechInferenceEngine,
)
from unitxt.system_prompts import TextualSystemPrompt

debug = False  # True for extra printing, set to False when commenting out max_test_instances below
max_test_instances = 20

# the available cards for the fleurs dataset, reflecting the target language, are:
#   de_de           German
#   es_419          Spanish, South America
#   fr_fr           French
#   it_it           Italian
#   ja_jp           Japanese
#   pt_br           Portuguese, Brazil
#   cmn_hans_cn     Chinese, Mandarin
test_dataset = load_dataset(  # select (un-comment) one of the test sets below
    # card="cards.fleurs.en_us.de_de",
    # card="cards.fleurs.en_us.es_419",
    # card="cards.fleurs.en_us.fr_fr",
    # card="cards.fleurs.en_us.it_it",
    # card="cards.fleurs.en_us.pt_br",
    card="cards.fleurs.en_us.ja_jp",
    # card="cards.fleurs.en_us.cmn_hans_cn",
    split="test",
    format="formats.chat_api",
    # max_test_instances=max_test_instances,  # comment out for running the entire test
    system_prompt=TextualSystemPrompt(
        text="Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
    ),
)

if debug:
    print(">>>>>>>>>>>>>>  test references  >>>>>>>>>>>>")
    for idx in range(max_test_instances):
        print(f">>>>>>   references {idx}:  ", test_dataset["references"][idx])

model = HFGraniteSpeechInferenceEngine(
    model_name="ibm-granite/granite-speech-3.3-8b",  # two options for Granite Speech 3.3:  2b  and  8b
    max_new_tokens=200,
)

predictions = model(test_dataset)

if debug:  # print translation reference texts for debug and inspection
    print(">>>>>>>>>>>>>>  model predictions  >>>>>>>>>>>>")
    for idx in range(max_test_instances):
        print(f">>>>>>>>>>> {idx}:   ", predictions[idx])

results = evaluate(
    predictions=predictions, data=test_dataset, calc_confidence_intervals=False
)

print("Global scores:")
print(results.global_scores.summary)
