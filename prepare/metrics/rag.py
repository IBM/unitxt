from src.unitxt import add_to_catalog
from src.unitxt.metrics import (
    BertScore,
    MetricPipeline,
    Reward,
    SentenceBert,
    TokenOverlap,
)
from src.unitxt.operators import CopyFields, ListFieldValues
from src.unitxt.test_utils.metrics import test_metric

bert_score_predictions = ["hello there general dude", "foo bar foobar"]
bert_score_references = [
    ["hello there general kenobi", "hello there!"],
    ["foo bar foobar", "foo bar"],
]
bert_score_instance_targets = [
    {"f1": 0.8, "precision": 0.86, "recall": 0.84, "score": 0.8, "score_name": "f1"},
    {"f1": 1.0, "precision": 1.0, "recall": 1.0, "score": 1.0, "score_name": "f1"},
]
bert_score_global_target = {
    "f1": 0.9,
    "f1_ci_high": 1.0,
    "f1_ci_low": 0.8,
    "precision": 0.93,
    "precision_ci_high": 1.0,
    "precision_ci_low": 0.86,
    "recall": 0.92,
    "recall_ci_high": 1.0,
    "recall_ci_low": 0.84,
    "score": 0.9,
    "score_ci_high": 1.0,
    "score_ci_low": 0.8,
    "score_name": "f1",
}


def main():
    metrics = {
        "metrics.token_overlap": TokenOverlap(),
        "metrics.bert_score.deberta.xlarge.mnli": BertScore(
            model_name="microsoft/deberta-xlarge-mnli"
        ),
        "metrics.sentence_bert.mpnet.base.v2": SentenceBert(
            model_name="sentence-transformers/all-mpnet-base-v2"
        ),
        "metrics.reward.deberta.v3.large.v2": Reward(
            model_name="OpenAssistant/reward-model-deberta-v3-large-v2"
        ),
    }

    predictions = ["apple", "boy", "cat"]
    references = [["apple2"], ["boys"], ["dogs"]]
    additional_inputs = [
        {"context": "apple 2e"},
        {"context": "boy"},
        {"context": "dog"},
    ]
    instance_targets = [
        {
            "f1": 0.67,
            "precision": 1.0,
            "recall": 0.5,
            "score": 0.67,
            "score_name": "f1",
        },
        {"f1": 1.0, "precision": 1.0, "recall": 1.0, "score": 1.0, "score_name": "f1"},
        {"f1": 0, "precision": 0, "recall": 0, "score": 0, "score_name": "f1"},
    ]

    # Currently, CopyFields does not delete the source fields,
    # so we get both "precision" and "precision_overlap_with_context" in results

    global_target = {
        "f1": 0.56,
        "f1_ci_high": 0.89,
        "f1_ci_low": 0.0,
        "f1_overlap_with_context": 0.56,
        "precision": 0.67,
        "precision_ci_high": 1.0,
        "precision_ci_low": 0.0,
        "precision_overlap_with_context": 0.67,
        "recall": 0.5,
        "recall_ci_high": 1.0,
        "recall_ci_low": 0.0,
        "recall_overlap_with_context": 0.5,
        "score": 0.56,
        "score_ci_high": 0.89,
        "score_ci_low": 0.0,
        "score_name": "f1",
    }

    metric = MetricPipeline(
        main_score="score",
        preprocess_steps=[
            CopyFields(
                field_to_field=[("additional_inputs/context", "references")],
                use_query=True,
            ),
            ListFieldValues(fields=["references"], to_field="references"),
        ],
        metric=metrics["metrics.token_overlap"],
        postpreprocess_steps=[
            CopyFields(
                field_to_field=[
                    ("score/global/f1", "score/global/f1_overlap_with_context"),
                    ("score/global/recall", "score/global/recall_overlap_with_context"),
                    (
                        "score/global/precision",
                        "score/global/precision_overlap_with_context",
                    ),
                ],
                use_query=True,
            ),
        ],
    )
    add_to_catalog(metric, "metrics.token_overlap_with_context", overwrite=True)

    test_metric(
        metric=metric,
        predictions=predictions,
        references=references,
        instance_targets=instance_targets,
        global_target=global_target,
        additional_inputs=additional_inputs,
    )

    metric = metrics["metrics.bert_score.deberta.xlarge.mnli"]

    test_metric(
        metric=metric,
        predictions=bert_score_predictions,
        references=bert_score_references,
        instance_targets=bert_score_instance_targets,
        global_target=bert_score_global_target,
    )

    metric = metrics["metrics.sentence_bert.mpnet.base.v2"]
    predictions = ["hello there general dude", "foo bar foobar"]
    references = [
        ["hello there general kenobi", "hello there!"],
        ["foo bar foobar", "foo bar"],
    ]

    instance_targets = [
        {"score": 0.71, "score_name": "score"},
        {"score": 1.0, "score_name": "score"},
    ]
    global_target = {
        "score": 0.86,
        "score_ci_high": 1.0,
        "score_ci_low": 0.71,
        "score_name": "score",
    }

    test_metric(
        metric=metric,
        predictions=predictions,
        references=references,
        instance_targets=instance_targets,
        global_target=global_target,
    )

    metric = metrics["metrics.reward.deberta.v3.large.v2"]
    predictions = ["hello there General Dude", "foo bar foobar"]
    references = [["How do you greet General Dude"], ["What is your name?"]]
    instance_targets = [
        {"label": "LABEL_0", "score": 0.14, "score_name": "score"},
        {"label": "LABEL_0", "score": 0.03, "score_name": "score"},
    ]
    global_target = {
        "score": 0.09,
        "score_ci_high": 0.14,
        "score_ci_low": 0.03,
        "score_name": "score",
    }

    test_metric(
        metric=metric,
        predictions=predictions,
        references=references,
        instance_targets=instance_targets,
        global_target=global_target,
    )

    for metric_id, metric in metrics.items():
        add_to_catalog(metric, metric_id, overwrite=True)

    # high level rag metrics
    # input fields
    # additional_inputs/question = the user question
    # additional_inputs/context_texts = the retrieved texts (list)
    # additional_inputs/context_text_ids = the ids of the retrieved texts (list)
    # additional_inputs/reference_texts = the retrieval reference texts (list)
    # additional_inputs/reference_text_ids = the ids of the retrieval reference texts (list)
    # additional_inputs/answer = the answer generated by the model
    # additional_inputs/reference_answers = the answer reference (list)

    # context_precision_unsupervised = MetricPipeline(
    #     main_score="score",
    #     preprocess_steps=[
    #         CopyFields(
    #             field_to_field=[
    #                 ("additional_inputs/context_texts", "additional_inputs/x"),
    #                 ("additional_inputs/question", "additional_inputs/y")], use_query=True
    #         ),
    #     ],
    #     metric=Perplexity(
    #         reduction_map={"max": ["perplexity"]},
    #         model_name="google/flan-t5-small",
    #         perplexity_prompt="Generate a question based on the given content:",
    #     ),
    # )
    #
    # context_precision_supervised = MetricPipeline(
    #     main_score="score",
    #     preprocess_steps=[
    #         CopyFields(
    #             field_to_field=[
    #                 ("additional_inputs/context_text_ids", "predictions"),
    #                 ("additional_inputs/reference_text_ids", "references")], use_query=True
    #         ),
    #     ],
    #     metric=MAP(),
    # )
    #
    # context_precision_unsupervised = MetricPipeline(
    #     main_score="score",
    #     preprocess_steps=[
    #         CopyFields(
    #             field_to_field=[
    #                 ("additional_inputs/context_texts", "predictions"),
    #                 ("additional_inputs/reference_text_ids", "references")], use_query=True
    #         ),
    #     ],
    #     metric=MRR(),
    # )
    #
    # add_to_catalog(context_relevance, "metrics.rag.context_relevance", overwrite=True)


if __name__ == "__main__":
    main()
