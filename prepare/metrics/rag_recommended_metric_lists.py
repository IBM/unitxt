from unitxt import add_to_catalog
from unitxt.metrics import MetricsList

recommended_metrics = {
    "cpu_only": {
        "answer_correctness": "token_recall",
        "faithfulness": "token_k_precision",
        "answer_relevance": "token_recall",
        "context_relevance": "token_precision",
        "context_correctness": "mrr",
    },
    "small_llm": {
        "answer_correctness": "bert_score_recall_ml",
        "faithfulness": "vectara_hhem_2_1",
        "answer_relevance": "answer_reward",
        "context_relevance": "sentence_bert_mini_lm",
        "context_correctness": "mrr",
    },
    "llmaj_watsonx": {
        "answer_correctness": "llama_3_3_70b_instruct_watsonx_judge",
        "faithfulness": "llama_3_3_70b_instruct_watsonx_judge",
        "answer_relevance": "llama_3_3_70b_instruct_watsonx_judge",
        "context_relevance": "llama_3_3_70b_instruct_watsonx_judge",
        "context_correctness": "mrr",
    },
    "llmaj_rits": {
        "answer_correctness": "llama_3_3_70b_instruct_rits_judge",
        "faithfulness": "llama_3_3_70b_instruct_rits_judge",
        "answer_relevance": "llama_3_3_70b_instruct_rits_judge",
        "context_relevance": "llama_3_3_70b_instruct_rits_judge",
        "context_correctness": "mrr",
    },
    "llmaj_azure": {
        "answer_correctness": "gpt_4o_azure_judge",
        "faithfulness": "gpt_4o_azure_judge",
        "answer_relevance": "gpt_4o_azure_judge",
        "context_relevance": "gpt_4o_azure_judge",
        "context_correctness": "mrr",
    },
}


def get_metrics_types_per_task(unitxt_task):
    metric_types = ["answer_correctness", "faithfulness", "answer_relevance"]
    if unitxt_task != "response_generation":
        metric_types.extend(["context_relevance", "context_correctness"])
    return metric_types


def get_recommended_metrics(resources_string, rag_unitxt_task):
    recommended_metrics_types_to_names = recommended_metrics[resources_string]
    metric_types = get_metrics_types_per_task(rag_unitxt_task)
    recommended_metrics_types_to_names = dict(
        filter(
            lambda x: x[0] in metric_types, recommended_metrics_types_to_names.items()
        )
    )
    return [
        f"metrics.rag_by_task.{rag_unitxt_task}.{k}.{v}"
        for k, v in recommended_metrics_types_to_names.items()
    ]


def register_recommended_metric_lists():
    for resource_str in recommended_metrics.keys():
        for rag_unitxt_task in ["response_generation", "end_to_end", "autorag"]:
            metrics = MetricsList(
                get_recommended_metrics(resource_str, rag_unitxt_task)
            )
            add_to_catalog(
                metrics,
                f"metrics.rag_by_task.{rag_unitxt_task}.recommended_{resource_str}.all",
                overwrite=True,
            )


register_recommended_metric_lists()
