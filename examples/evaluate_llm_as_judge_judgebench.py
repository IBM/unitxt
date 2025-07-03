import json

from unitxt.api import evaluate, load_dataset
from unitxt.inference import MetricInferenceEngine

cards = [
    "cards.judge_bench.newswoom.relevance",
    "cards.judge_bench.newswoom.coherence",
    "cards.judge_bench.newswoom.fluency",
    "cards.judge_bench.newswoom.informativeness",
    "cards.judge_bench.newswoom.relevance",
    "cards.judge_bench.roscoe.cosmos.overall.contradiction",
    "cards.judge_bench.roscoe.cosmos.overall.coherence",
    "cards.judge_bench.roscoe.cosmos.overall.missing_steps",
    "cards.judge_bench.roscoe.cosmos.overall.overall_quality",
    "cards.judge_bench.toxic_chat.jailbreaking",
    "cards.judge_bench.toxic_chat.toxicity",
    "cards.judge_bench.dices.safety",
    "cards.judge_bench.inferential_strategies.sound_reasoning",
    "cards.judge_bench.cola.grammaticality",
]


for card in cards:
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
