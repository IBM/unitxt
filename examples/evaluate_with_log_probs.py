from unitxt import infer
from unitxt.inference_engines import (
    HFLogProbScoringEngine,
    SelectingByScoreInferenceEngine,
)
from unitxt.text_utils import print_dict

dataset = infer(
    [
        {
            "question": "What is the capital of Texas?",
            "choices": ["Austin", "Tel Aviv"],
        },
        {"question": "What is the color of the sky?", "choices": ["Blue", "Red"]},
    ],
    engine=SelectingByScoreInferenceEngine(
        scorer_engine=HFLogProbScoringEngine(model_name="gpt2", batch_size=1)
    ),
    task="tasks.qa.multiple_choice.open",
    template="templates.qa.multiple_choice.title",
    return_data=True,
)

for instance in dataset:
    print_dict(instance, keys_to_print=["source", "prediction"])
