from unitxt import add_to_catalog
from unitxt.metrics import (
    MetricPipeline,
)
from unitxt.operators import Copy, ListFieldValues

task_names = ["external_rag", "response_generation", "end_to_end"]
base = "metrics.rag"


def get_preprocess_steps(task):
    # This metric compares the answer (as the prediction) to the question (as the reference).
    # We have to wrap the question by a list (otherwise it will be a string),
    # because references are expected to be lists
    last_step = ListFieldValues(fields=["references"], to_field="references")
    if task == "external_rag":
        return [
            Copy(
                field_to_field={
                    "question": "references",
                    "answer": "prediction",
                },
            ),
            last_step,
        ]
    if task == "response_generation":
        return [
            Copy(
                field_to_field={
                    "task_data/question": "references",
                }
            ),
            last_step,
        ]
    if task == "end_to_end":
        return [
            Copy(
                field_to_field={
                    "task_data/question": "references",
                    "prediction/answer": "prediction",
                }
            ),
            last_step,
        ]
    raise ValueError(f"Unsupported rag task {task}")


for task in task_names:
    answer_reward = MetricPipeline(
        main_score="reward_score",
        preprocess_steps=get_preprocess_steps(task),
        metric="metrics.reward.deberta_v3_large_v2",
        score_prefix="answer_relevance_",
    )
    add_to_catalog(
        answer_reward, f"{base}.{task}.answer_relevance.answer_reward", overwrite=True
    )
    if task == "external_rag":
        add_to_catalog(answer_reward, f"{base}.{task}.answer_reward", overwrite=True)

    answer_token_overlap = MetricPipeline(
        main_score="recall",
        preprocess_steps=get_preprocess_steps(task),
        metric="metrics.token_overlap",
        score_prefix="answer_relevance_token_recall_",
    )
    add_to_catalog(
        answer_token_overlap,
        f"{base}.{task}.answer_relevance.token_recall",
        overwrite=True,
    )
#
# answer_inference = MetricPipeline(
#     main_score="perplexity",
#     preprocess_steps=[
#         Copy(
#             field_to_field={"task_data/contexts": "references", "answer": "prediction"},
#             not_exist_do_nothing=True,
#         ),
#         Copy(field_to_field={"contexts": "references"}, not_exist_do_nothing=True),
#     ],
#     metric="metrics.perplexity_nli.t5_nli_mixture",
# )
# add_to_catalog(answer_inference, "metrics.rag.answer_inference", overwrite=True)
