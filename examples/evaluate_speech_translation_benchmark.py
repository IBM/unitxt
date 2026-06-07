# this python script shows an example of running speech translation benchmark evaluation for Granite Speech
# using the Fleurs and Covost2 datasets

# to run on a single test set use subset=... below; the list of subsets is:
# fleurs_en_de, fleurs_en_es, fleurs_en_fr, fleurs_en_it, fleurs_en_ja, fleurs_en_pt, fleurs_en_pt,
# covost2_en_de, covost2_en_ja, covost2_de_en, covost2_es_en, covost2_fr_en, covost2_pt_en

from unitxt import evaluate, load_dataset
from unitxt.inference import (
    HFGraniteSpeechInferenceEngine,
)
from unitxt.system_prompts import TextualSystemPrompt

dataset = load_dataset(
    "benchmarks.speech_translation",
    # max_samples_per_subset=100,   # while this is commented out, the entire test set is used
    # subset="fleurs_en_fr",        #to run only a single test set
    system_prompt=TextualSystemPrompt(
        text="Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
    ),
    split="test",
)

model = HFGraniteSpeechInferenceEngine(
    model_name="ibm-granite/granite-speech-3.3-8b",  # two options for Granite Speech 3.3:  2b  and  8b
    max_new_tokens=200,
)

predictions = model(dataset)
results = evaluate(
    predictions=predictions, data=dataset, calc_confidence_intervals=False
)

print("Global scores:")
print(results.global_scores.summary)

print("Subsets scores:")
print(results.subsets_scores.summary)
