from unitxt import add_to_catalog
from unitxt.collections_operators import Wrap
from unitxt.metrics import (
    MetricPipeline,
)
from unitxt.operators import Copy
from unitxt.serializers import MultiTypeSerializer

task_names = ["external_rag", "response_generation", "end_to_end"]
base = "metrics.rag"


def get_preprocess_steps(task):
    # This metric compares the answer (as the prediction) to the question (as the reference).
    # We have to wrap the question by a list (otherwise it will be a string),
    # because references are expected to be lists
    if task == "external_rag":
        return [
            Copy(field="prediction/answer", to_field="prediction"),
            Wrap(field="task_data/question", inside="list", to_field="references"),
        ]
    if task == "response_generation":
        return [
            Wrap(field="task_data/question", inside="list", to_field="references"),
            MultiTypeSerializer(field="references", process_every_value=True),
        ]
    if task == "end_to_end":
        return [
            Copy(field="prediction/answer", to_field="prediction"),
            Wrap(field="task_data/question", inside="list", to_field="references"),
            MultiTypeSerializer(field="references", process_every_value=True),
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
