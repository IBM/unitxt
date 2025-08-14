# this python script shows an example of running speech recognition evaluation for Granite Speech using the Hugging Face ESB datasets and the CommonVoice datasets

from unitxt import evaluate, load_dataset
from unitxt.inference import (
    CrossProviderInferenceEngine,
    HFGraniteSpeechInferenceEngine,
)
from unitxt.system_prompts import TextualSystemPrompt

USE_RITS = True  #  whether to use RITS service or running the model locally

test_dataset = load_dataset(
    # select (uncomment) only one of the following cards (datasets)
    # for evaluating a benchmark with multiple cards - see evaluate_speech_recognition_benchmark.py in the same directory (examples)
    card="cards.esb.ami",
    # card="cards.esb.voxpopuli",
    # card="cards.esb.librispeech",
    # card="cards.esb.spgispeech",
    # card="cards.esb.earnings22",
    # card="cards.esb.tedlium",
    # card="cards.commonvoice.en"
    # card="cards.commonvoice.de"
    # card="cards.commonvoice.fr"
    # card="cards.commonvoice.es"
    # card="cards.commonvoice.pt"
    split="test",
    format="formats.chat_api",
    max_test_instances=5,  # to tun limited part of the test set
    system_prompt=TextualSystemPrompt(
        text="Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
    ),
)

if not USE_RITS:
    # locally running the model, it needs GPU to run properly
    model = HFGraniteSpeechInferenceEngine(
        model_name="ibm-granite/granite-speech-3.3-8b",  # two options for Granite Speech 3.3:  2b  and  8b
        max_new_tokens=120,  # 200 for 2b,  120 for 8b
    )
if USE_RITS:
    # using the RITS remote service for inferencing
    model = CrossProviderInferenceEngine(
        model="granite-speech-3-3-8b",  # in RITS only the 8b version of Granite Speech is available
        provider="rits",
        # provider_specific_args={"rits": {"max_new_tokens": 120}},
        max_new_tokens=120,
    )

predictions = model(test_dataset)
results = evaluate(
    predictions=predictions, data=test_dataset, calc_confidence_intervals=False
)

print("Global scores:")
print(results.global_scores.summary)
