import json
from typing import Any

import numpy as np
from numpy import mean
from unitxt.api import evaluate, load_dataset
from unitxt.blocks import (
    MapInstanceValues,
    Rename,
    Set,
    SplitRandomMix,
    TaskCard,
)
from unitxt.llm_as_judge_constants import DirectCriteriaCatalogEnum
from unitxt.llm_as_judge_operators import LoadCriteriaWithOptions
from unitxt.loaders import LoadFromAPI, LoadFromDictionary
from unitxt.task import Task
from unitxt.templates import NullTemplate

evaluators = [
    "watsonx.llama3_1_70b",
    "watsonx.llama3_1_8b",
    "watsonx.granite3_0_8b",
    "watsonx.mixtral_large",
]

params = "[criteria_field=criteria,include_prompts_in_result=True,context_fields=None,check_positional_bias=False,response_variable_name=text"
total_instances = 200
card = TaskCard(
    loader=LoadFromAPI(
        urls={"test":"https://raw.githubusercontent.com/dmg-illc/JUDGE-BENCH/refs/heads/master/data/toxic_chat/toxic_chat_test.json"},
        data_classification_policy=["public"],
        data_field="instances",
        headers={
            "Authorization": ""
        },
        loader_limit=total_instances,
    ),
    preprocess_steps=[
        SplitRandomMix(
            mix={
                "test": "test[100%]",
            }
        ),
        Rename(field="instance", to_field="text"),
        Rename(field="annotations/toxicity/majority_human", to_field="ground_truth_toxicity"),
        # Rename(field="annotations/jailbreaking/majority_human", to_field="ground_truth_jailbreak"),
        Set(fields={"criteria": "metrics.llm_as_judge.direct.criteria.toxicity"}),
        # Set(fields={"jailbreak_criteria": "metrics.llm_as_judge.direct.criteria.user_message_jailbreak"}),
        MapInstanceValues(mappers={
            "ground_truth_toxicity": {
                "0": "non-toxic",
                "1": "toxic"
            },
            # "ground_truth_jailbreak": {
            #     "0": "No",
            #     "1": "Yes"
            # }
        }),
        LoadCriteriaWithOptions(field="criteria")
    ],
    task=Task(
        default_template=NullTemplate(),
        input_fields={},
        reference_fields={"criteria": Any, "ground_truth_toxicity": str, "text": str},
        prediction_type=str,
        metrics=[f"metrics.llm_as_judge.direct.{evaluator}{params},score_prefix={i}_]" for i, evaluator in enumerate(evaluators)],
    ),
)

dataset = load_dataset(card=card, template="templates.empty", split="test")

predictions = [json.loads(instance["task_data"])["text"] for instance in dataset]

results = evaluate(predictions=predictions, data=dataset)

all_predictions = [{k.split("toxicity_using_")[1]:v for k,v in instance["score"]["instance"].items() if "toxicity_using_" in k} for instance in results]

data = {"test": [{"ground_truth_toxicity": float(DirectCriteriaCatalogEnum.TOXICITY.value.option_map[instance["task_data"]["ground_truth_toxicity"]])} for instance in results]}

def get_score_index_from_evaluator_name(result, evaluator_name):
    return [c for c in result["score"]["instance"].keys() if evaluator_name in c][0].split("_")[0]


data_to_inspect = {
    k: [{
            "ground": ground,
            "pred": pred,
            "text": json.loads(instance["task_data"])["text"],
            "assessment": result["score"]["instance"][f"{get_score_index_from_evaluator_name(result, k)}_toxicity_assessment"],
            "prompts": result["score"]["instance"][f"{get_score_index_from_evaluator_name(result, k)}_toxicity_prompts"],
            "completion": result["score"]["instance"][f"{get_score_index_from_evaluator_name(result, k)}_toxicity_option_selection_completion"],
        } for i, (ground, pred, result, instance) in enumerate(
        zip(
            [d["ground_truth_toxicity"] for d in data["test"]],
            [p[k] for p in all_predictions],
            results,
            dataset
        )) if ground != pred] for k in all_predictions[0].keys()}


benchmark_results = {}

for key in all_predictions[0]:
    card = TaskCard(
        loader=LoadFromDictionary(data=data, data_classification_policy=["public"]),
        task= Task(
            input_fields={},
            reference_fields={"ground_truth_toxicity": float},
            prediction_type=float,
            metrics=["metrics.spearman"],
        )
    )

    dataset = load_dataset(
        card=card,
        template="templates.empty[postprocessors=[processors.cast_to_float_return_nan_if_failed]]",
        split="test")

    predictions = [instance[key] for instance in all_predictions]
    results = evaluate(predictions=predictions, data=dataset)
    benchmark_results[key] = {"pearson": results.global_scores["score"]}
    benchmark_results[key]["data_to_inspect"] = data_to_inspect[key]
    benchmark_results[key]["agreement"] = mean(np.asarray([d["ground_truth_toxicity"] for d in data["test"]]) == np.asarray([p[key] for p in all_predictions]))

with open(f"toxic_chat_benchmark_results_{total_instances}_instances.json", "w") as f:
    json.dump(benchmark_results, fp=f, indent=4)
