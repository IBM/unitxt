from unitxt import evaluate, load_dataset
from unitxt.inference import (
    HFGraniteSpeechInferenceEngine,
)
from unitxt.system_prompts import TextualSystemPrompt

test_dataset = load_dataset(
    card="cards.ami",
    split="test",
    format="formats.chat_api",
    max_test_instances=10,
    system_prompt=TextualSystemPrompt(
        text="Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
    ),
)

model = HFGraniteSpeechInferenceEngine(
    model_name="ibm-granite/granite-speech-3.3-2b",
    max_new_tokens=200,
)

predictions = model(test_dataset)
results = evaluate(predictions=predictions, data=test_dataset)

print("Global scores:")
print(results.global_scores.summary)
