from unitxt import evaluate, load_dataset
from unitxt.inference import (
    HFGraniteSpeechInferenceEngine,
)
from unitxt.system_prompts import TextualSystemPrompt

dataset = load_dataset(
    "benchmarks.speech_recognition",
    max_samples_per_subset=5,
    subset="ami",
    system_prompt=TextualSystemPrompt(
        text="Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
    ),
    split="test",
)

model = HFGraniteSpeechInferenceEngine(
    model_name="ibm-granite/granite-speech-3.3-2b",
    max_new_tokens=200,
)

predictions = model(dataset)
results = evaluate(predictions=predictions, data=dataset)

print("Global scores:")
print(results.global_scores.summary)

print("Subsets scores:")
print(results.subsets_scores.summary)
