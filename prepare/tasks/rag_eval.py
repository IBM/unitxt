from unitxt import add_to_catalog
from unitxt.task import Task

rag_classification_metrics = {
    "binary": [
        "metrics.spearman",
        "metrics.kendalltau_b",
        "metrics.roc_auc",
        "metrics.f1_binary",
        "metrics.accuracy_binary",
        "metrics.precision_binary",
        "metrics.recall_binary",
        "metrics.max_f1_binary",
        "metrics.max_accuracy_binary",
    ],
    "non_binary": ["metrics.spearman", "metrics.kendalltau_b"],
}

field_to_type = {
    "answer": "str",
    "ground_truths": "List[str]",
    "question": "str",
    "choices": "List[str]",
    "contexts": "List[str]",
    "number_val": "Union[float, int]",
    "is_correct": "List[str]",
    "is_faithful": "List[str]",
    "is_context_relevant": "List[str]",
    "is_relevant": "List[str]",
}


def convert_to_dict_of_type(field_list):
    return {k: field_to_type[k] for k in field_list}


for binary_val in rag_classification_metrics:
    add_to_catalog(
        Task(
            inputs=convert_to_dict_of_type(
                ["answer", "ground_truths", "question", "choices"]
            ),
            outputs=convert_to_dict_of_type(["is_correct", "number_val"]),
            metrics=rag_classification_metrics[binary_val],
            prediction_type="float",
        ),
        f"tasks.rag_eval.correctness.{binary_val}",
    )

    add_to_catalog(
        Task(
            inputs=convert_to_dict_of_type(
                ["answer", "contexts", "question", "choices"]
            ),
            outputs=convert_to_dict_of_type(["is_faithful", "number_val"]),
            metrics=rag_classification_metrics[binary_val],
            prediction_type="float",
        ),
        f"tasks.rag_eval.faithfulness.{binary_val}",
    )

    add_to_catalog(
        Task(
            inputs=convert_to_dict_of_type(["contexts", "question", "choices"]),
            outputs=convert_to_dict_of_type(["is_context_relevant", "number_val"]),
            metrics=rag_classification_metrics[binary_val],
            prediction_type="float",
        ),
        f"tasks.rag_eval.context_relevance.{binary_val}",
    )

    add_to_catalog(
        Task(
            inputs=convert_to_dict_of_type(["answer", "question", "choices"]),
            outputs=convert_to_dict_of_type(["is_relevant", "number_val"]),
            metrics=rag_classification_metrics[binary_val],
            prediction_type="float",
        ),
        f"tasks.rag_eval.answer_relevance.{binary_val}",
    )

    add_to_catalog(
        Task(
            inputs=convert_to_dict_of_type(
                ["answer", "contexts", "question", "choices"]
            ),
            outputs=convert_to_dict_of_type(["is_correct", "number_val"]),
            metrics=rag_classification_metrics[binary_val],
            prediction_type="float",
        ),
        f"tasks.rag_eval.correctness_holistic.{binary_val}",
    )

    add_to_catalog(
        Task(
            inputs=convert_to_dict_of_type(
                ["answer", "question", "choices", "ground_truths", "contexts"]
            ),
            outputs=convert_to_dict_of_type(["is_correct", "number_val"]),
            metrics=rag_classification_metrics[binary_val],
            prediction_type="float",
        ),
        f"tasks.rag_eval.correctness_reference_based.{binary_val}",
    )
