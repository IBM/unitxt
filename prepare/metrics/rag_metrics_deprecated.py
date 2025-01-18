from unitxt import add_to_catalog
from unitxt.collections_operators import Wrap
from unitxt.metrics import MetricPipeline
from unitxt.operators import Copy, ListFieldValues

base = "metrics.rag"
new_base = "metrics.rag.autorag"


def add_metric_pipeline_to_catalog(
    metric_main_score,
    metric_preprocess_steps,
    orig_metric_catalog_name,
    metric_dimension,
    metric_new_catalog_name="",
):
    metric_path = f"{base}.{metric_dimension}.{metric_new_catalog_name}".strip(".")
    new_metric_path = get_replacing_metric(metric_path)
    metric = MetricPipeline(
        main_score=metric_main_score,
        preprocess_steps=metric_preprocess_steps,
        metric=orig_metric_catalog_name,
        __deprecated_msg__=f"This metric should be replaced with {new_metric_path}",
    )
    add_to_catalog(metric, metric_path, overwrite=True)


def add_to_catalog_with_default(
    metric_main_score,
    metric_preprocess_steps,
    orig_metric_catalog_name,
    metric_dimension,
    metric_new_catalog_name,
    default_metric_name,
):
    add_metric_pipeline_to_catalog(
        metric_main_score,
        metric_preprocess_steps,
        orig_metric_catalog_name,
        metric_dimension,
        metric_new_catalog_name,
    )
    if new_catalog_name == default_metric_name:
        add_metric_pipeline_to_catalog(
            metric_main_score,
            metric_preprocess_steps,
            orig_metric_catalog_name,
            metric_dimension,
        )


def get_replacing_metric(depr_metric):
    return depr_metric.replace(base, new_base)


default_ac = "token_recall"
preprocess_steps = [
    Copy(
        field_to_field={
            "task_data/reference_answers": "references",
            "answer": "prediction",
        },
        not_exist_do_nothing=True,
    ),
    Copy(
        field_to_field={"ground_truths": "references"},
        not_exist_do_nothing=True,
    ),
]
for new_catalog_name, base_catalog_name, main_score in [
    ("token_recall", "metrics.token_overlap", "recall"),
    ("bert_score_recall", "metrics.bert_score.deberta_large_mnli", "recall"),
    (
        "bert_score_recall_ml",
        "metrics.bert_score.deberta_v3_base_mnli_xnli_ml",
        "recall",
    ),
    ("sentence_bert_bge", "metrics.sentence_bert.bge_large_en_1_5", "sbert_score"),
    ("sentence_bert_mini_lm", "metrics.sentence_bert.minilm_l12_v2", "sbert_score"),
]:
    add_to_catalog_with_default(
        main_score,
        preprocess_steps,
        base_catalog_name,
        "answer_correctness",
        new_catalog_name,
        default_ac,
    )


##################
# answer_relevance
##################
answer_relevance_preprocess_steps = preprocess_steps = [
    Copy(
        field_to_field={"task_data/question": "references", "answer": "prediction"},
        not_exist_do_nothing=True,
    ),
    Copy(field_to_field={"question": "references"}, not_exist_do_nothing=True),
    # This metric compares the answer (as the prediction) to the question (as the reference).
    # We have to wrap the question by a list (otherwise it will be a string),
    # because references are expected to be lists
    ListFieldValues(fields=["references"], to_field="references"),
]
add_metric_pipeline_to_catalog(
    "score", preprocess_steps, "metrics.reward.deberta_v3_large_v2", "answer_reward"
)
add_metric_pipeline_to_catalog(
    "recall",
    preprocess_steps,
    "metrics.token_overlap",
    "answer_relevance",
    "token_recall",
)

answer_inference = MetricPipeline(
    main_score="perplexity",
    preprocess_steps=[
        Copy(
            field_to_field={"task_data/contexts": "references", "answer": "prediction"},
            not_exist_do_nothing=True,
        ),
        Copy(field_to_field={"contexts": "references"}, not_exist_do_nothing=True),
    ],
    metric="metrics.perplexity_nli.t5_nli_mixture",
    __deprecated_msg__="This metric is deprecated",
)
add_to_catalog(answer_inference, "metrics.rag.answer_inference", overwrite=True)


#####################
# context correctness
#####################
default_context_correctness = "mrr"
preprocess_steps = [
    Copy(field="context_ids", to_field="prediction"),
    Wrap(field="ground_truths_context_ids", inside="list", to_field="references"),
]
for new_catalog_name, base_catalog_name, main_score in [
    ("mrr", "metrics.mrr", "mrr"),
    ("map", "metrics.map", "map"),
    ("retrieval_at_k", "metrics.retrieval_at_k", "match_at_1"),
]:
    add_to_catalog_with_default(
        main_score,
        preprocess_steps,
        base_catalog_name,
        "context_correctness",
        new_catalog_name,
        default_context_correctness,
    )


####################
# Context Relevance
####################
default_context_relevance = "perplexity_flan_t5_small"
preprocess_steps = [
    Copy(
        field_to_field={
            "task_data/contexts": "references",
            "question": "prediction",
        },
        not_exist_do_nothing=True,
    ),
    Copy(field_to_field={"contexts": "references"}, not_exist_do_nothing=True),
]
for new_catalog_name, base_catalog_name, main_score in [
    ("perplexity_flan_t5_small", "metrics.perplexity_q.flan_t5_small", "perplexity"),
    ("sentence_bert_bge", "metrics.sentence_bert.bge_large_en_1_5", "sbert_score"),
    ("sentence_bert_mini_lm", "metrics.sentence_bert.minilm_l12_v2", "sbert_score"),
    ("token_precision", "metrics.token_overlap", "precision"),
]:
    add_to_catalog_with_default(
        main_score,
        preprocess_steps,
        base_catalog_name,
        "context_relevance",
        new_catalog_name,
        default_context_relevance,
    )


##############
# faithfulness
##############
default_faithfulness = "token_k_precision"
preprocess_steps = [
    Copy(
        field_to_field={
            "task_data/contexts": "references",
            "answer": "prediction",
        },
        not_exist_do_nothing=True,
    ),
    Copy(field_to_field={"contexts": "references"}, not_exist_do_nothing=True),
]
for new_catalog_name, base_catalog_name, main_score in [
    ("token_k_precision", "metrics.token_overlap", "precision"),
    ("bert_score_k_precision", "metrics.bert_score.deberta_large_mnli", "precision"),
    (
        "bert_score_k_precision_ml",
        "metrics.bert_score.deberta_v3_base_mnli_xnli_ml",
        "precision",
    ),
    ("sentence_bert_bge", "metrics.sentence_bert.bge_large_en_1_5", "sbert_score"),
    ("sentence_bert_mini_lm", "metrics.sentence_bert.minilm_l12_v2", "sbert_score"),
    ("vectara_hhem_2_1", "metrics.vectara_groundedness_hhem_2_1", "hhem_score"),
]:
    add_to_catalog_with_default(
        main_score,
        preprocess_steps,
        base_catalog_name,
        "faithfulness",
        new_catalog_name,
        default_faithfulness,
    )
