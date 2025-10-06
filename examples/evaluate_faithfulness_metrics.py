import json

import pandas as pd
import unitxt
from unitxt.api import evaluate, load_dataset
from unitxt.benchmark import Benchmark
from unitxt.inference import MetricInferenceEngine
from unitxt.standard import DatasetRecipe
from unitxt.templates import InputOutputTemplate

unitxt.settings.allow_unverified_code = True
unitxt.settings.dataset_cache_default = True

card_subsets = [
    "covidqa",
    "cuad",
    "delucionqa",
    "emanual",
    "expertqa",
    "finqa",
    "hagrid",
    "hotpotqa",
    "msmarco",
    "pubmedqa",
    "tatqa",
    #                "techqa" # Fails due to bad char in text
]

# card_subsets = ["covidqa"]
card = "cards.rag_eval.faithfulness.ragbench"

template = InputOutputTemplate(
    output_format="{number_val}",
    input_format="{question}",  # "CONTEXTS:{contexts}\n\n\n\QUESTION:{question}\n\n\nANSWER:{answer}",
    postprocessors=["processors.cast_to_float_return_0_5_if_failed"],
)

subsets = {
    card_subset: DatasetRecipe(
        card=f"{card}.{card_subset}",
        template=template,
        metrics=[
            "metrics.f1_binary",
            "metrics.f1_binary[average=macro,score_prefix=macro_]",
        ],
    )
    for card_subset in card_subsets
}

benchmark = Benchmark(
    format="formats.empty",
    max_samples_per_subset=20,
    loader_limit=300,
    subsets=subsets,
)

dataset = load_dataset(
    benchmark,
    split="test",
)
for instance in dataset:
    task_data = json.loads(instance["task_data"])


metrics_to_score_names = {}

criterion = "metrics.llm_as_judge.direct.criteria.reference_document_faithfulness"
llm_as_judge_metric = f"metrics.llm_as_judge.direct.rits.llama3_3_70b[check_positional_bias=False,criteria={criterion}, context_fields=[contexts,question]]"
llm_score_name = "reference_document_faithfulness"
metrics_to_score_names[llm_as_judge_metric] = llm_score_name

llm_as_judge_metric = f"metrics.llm_as_judge.evalassist.direct.rits.llama3_3_70b[criteria={criterion},context_fields=[contexts,question]]"
metrics_to_score_names[llm_as_judge_metric] = llm_score_name

llm_as_judge_metric = (
    "metrics.rag.external_rag.faithfulness.llama_3_3_70b_instruct_watsonx_judge"
)
llm_score_name = "faithfulness_judge"
metrics_to_score_names[llm_as_judge_metric] = llm_score_name
metrics_to_score_names["all_one"] = "score"
df = pd.DataFrame(
    columns=[
        "metric",
        "f1_macro",
        "f1_faithful",
        "f1_not_faithful",
        "num_of_instances",
    ]
)

for metric, score_name in metrics_to_score_names.items():
    # print(json.dumps(task_data,indent=4))
    # print(json.dumps(instance,indent=4))
    # print(instance["references"])

    if metric == "all_one":
        new_predictions = [1.0] * len(dataset)
    else:
        model = MetricInferenceEngine(metric=metric, prediction_field="answer")
        predictions = model(dataset)
        new_predictions = [prediction[score_name] for prediction in predictions]
    results = evaluate(predictions=new_predictions, data=dataset)

    sums = {}
    counts = {}

    for _, inner_dict in results.subsets_scores.items():
        if isinstance(inner_dict, dict):
            for key, value in inner_dict.items():
                if isinstance(value, float):
                    sums[key] = sums.get(key, 0) + value
                    counts[key] = counts.get(key, 0) + 1
    #
    averages = {key: sums[key] / counts[key] for key in sums}

    df.loc[len(df)] = [
        str(metric),
        averages["macro_f1_binary"],
        averages["f1_binary"],
        averages["f1_binary_neg"],
        results.global_scores["num_of_instances"],
    ]

    print("Instance Results:")
    print(results.instance_scores.summary)

    print("Subsets Results (details):")
    print(results.subsets_scores)

    print("Subsets Results :")
    print(results.subsets_scores.summary)

    df = df.round(decimals=2)
    print(df.to_markdown())
