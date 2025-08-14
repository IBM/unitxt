# this python script shows an example of running speech recognition benchmark evaluation for Granite Speech
# using the Hugging Face ESB datasets (English) and the multilingial CommonVoice datasets

# to run on a single test set use subset=... below; the list of subsets is:
# voxpopuli, ami, librispeech, spgispeech, tedlium, earnings22,
# commonvoice_en, commonvoice_de, commonvoice_es, commonvoice_fr, commonvoice_pt

from unitxt import evaluate, load_dataset
from unitxt.inference import (
    HFGraniteSpeechInferenceEngine,
)
from unitxt.system_prompts import TextualSystemPrompt

dataset = load_dataset(
    "benchmarks.speech_recognition",
    max_samples_per_subset=5,  # while this is commented out, the entire test set is used
    # subset="ami",   #to tun only a single dataset
    system_prompt=TextualSystemPrompt(
        text="Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
    ),
    split="test",
)

model = HFGraniteSpeechInferenceEngine(
    model_name="ibm-granite/granite-speech-3.3-2b",  # two options for Granite Speech 3.3:  2b  and  8b
    max_new_tokens=200,  # 200 for 2b,  120 for 8b
)

predictions = model(dataset)
results = evaluate(predictions=predictions, data=dataset)

print("Global scores:")
print(results.global_scores.summary)

print("Subsets scores:")
print(results.subsets_scores.summary)
