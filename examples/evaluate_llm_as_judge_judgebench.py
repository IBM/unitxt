import json
import os

from unitxt.api import evaluate, load_dataset
from unitxt.inference import MetricInferenceEngine
from unitxt.settings_utils import get_constants


def get_judgebench_cards():
    constants = get_constants()
    judgebench_dir = os.path.join(
        constants.catalog_dir,
        "cards",
        "judge_bench",
    )

    judgebench_cards = []

    for dirpath, _, filenames in os.walk(judgebench_dir):
        for file in filenames:
            if file.endswith(".json"):
                # Get the relative path without the .json extension
                relative_path = os.path.relpath(
                    os.path.join(dirpath, file), judgebench_dir
                )
                without_extension = os.path.splitext(relative_path)[0]
                dotted_path = without_extension.replace(os.path.sep, ".")
                judgebench_cards.append(f"cards.judge_bench.{dotted_path}")

    return judgebench_cards


cards = get_judgebench_cards()

for card in [cards[0]]:
    print("Running card ", card)
    dataset = load_dataset(card=card, split="test", loader_limit=3)

    params = (
        "[criteria_field=criteria,context_fields=None,include_prompts_in_result=true]"
    )

    metric_inference_engine = MetricInferenceEngine(
        metric=f"metrics.llm_as_judge.direct.watsonx.llama3_3_70b{params}",
    )

    predictions = metric_inference_engine.infer(dataset)
    criteria_name = json.loads(
        predictions[0][
            next(iter([key for key in predictions[0] if key.endswith("_criteria")]))
        ]
    )["name"]

    parsed_predictions = [p[criteria_name] for p in predictions]

    results = evaluate(predictions=parsed_predictions, data=dataset)

    print(results.global_scores.summary)
