from unitxt import add_to_catalog
from unitxt.metrics import (
    MetricPipeline,
)
from unitxt.operators import Copy

base = "metrics.rag_by_task"
tasks = ["autorag", "end_to_end"]
default = "perplexity_flan_t5_small"
dimension = "context_relevance"


def get_preprocess_steps(task):
    if task == "autorag":
        return [
            Copy(field="contexts", to_field="references"),
            Copy(field="question", to_field="prediction"),
        ]
    if task == "end_to_end":
        return [
            [
                Copy(field="prediction/contexts", to_field="references"),
                Copy(field="task_data/question", to_field="prediction"),
            ],
        ]
    raise ValueError(f"Unsupported rag task for {dimension}:{task}")


for task in tasks:
    for new_catalog_name, base_catalog_name, main_score in [
        (
            "perplexity_flan_t5_small",
            "metrics.perplexity_q.flan_t5_small",
            "perplexity",
        ),
        ("sentence_bert_bge", "metrics.sentence_bert.bge_large_en_1_5", "sbert_score"),
        ("sentence_bert_mini_lm", "metrics.sentence_bert.minilm_l12_v2", "sbert_score"),
        ("token_precision", "metrics.token_overlap", "precision"),
    ]:
        metric = MetricPipeline(
            main_score=main_score,
            preprocess_steps=get_preprocess_steps(task).copy(),
            metric=base_catalog_name,
            score_prefix=f"{dimension}_{new_catalog_name}_",
        )
        add_to_catalog(
            metric, f"{base}.{task}.{dimension}.{new_catalog_name}", overwrite=True
        )

        if new_catalog_name == default:
            metric = MetricPipeline(
                main_score=main_score,
                preprocess_steps=get_preprocess_steps(task).copy(),
                metric=base_catalog_name,
                score_prefix=f"{dimension}_",
            )
            add_to_catalog(metric, f"{base}.{task}.{dimension}", overwrite=True)
#
# context_perplexity = MetricPipeline(
#     main_score="score",
#     preprocess_steps=get_test_pipeline_task_preprocess_steps(task).copy(),
#     metric="metrics.perplexity_q.flan_t5_small",
#     postprocess_steps=[
#         Copy(field="score/instance/reference_scores", to_field="score/instance/score")
#     ],
# )
# add_to_catalog(context_perplexity, "metrics.rag.context_perplexity", overwrite=True)
