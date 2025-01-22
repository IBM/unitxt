import statistics

from unitxt import get_logger, get_settings, load_dataset
from unitxt.api import evaluate
from unitxt.inference import (
    CrossProviderInferenceEngine,
)

logger = get_logger()
settings = get_settings()

# Use the HF load_dataset API, to load the squad QA dataset using the standard template in the catalog.
# We set loader_limit to 20 to reduce download time.
criterias = ["answer_relevance", "coherence", "conciseness"]
metrics = [
    "metrics.llm_as_judge.direct.rits.llama3_1_70b"
    "[context_fields=[context,question],"
    f"criteria=metrics.llm_as_judge.direct.criterias.{criteria}]"
    for criteria in criterias
]
dataset = load_dataset(
    card="cards.squad",
    metrics=metrics,
    loader_limit=20,
    max_test_instances=20,
    split="test",
)

# Infer a model to get predictions.
inference_model = CrossProviderInferenceEngine(
    model="llama-3-2-1b-instruct", provider="watsonx"
)

"""
We are using a CrossProviderInferenceEngine inference engine that supply api access to provider such as:
watsonx, bam, openai, azure, aws and more.

For the arguments these inference engines can receive, please refer to the classes documentation or read
about the the open ai api arguments the CrossProviderInferenceEngine follows.
"""
predictions = inference_model.infer(dataset)

gold_answers = [d[0] for d in dataset["references"]]

# Evaluate the predictions using the defined metric.
evaluated_predictions = evaluate(predictions=predictions, data=dataset)
evaluated_gold_answers = evaluate(predictions=gold_answers, data=dataset)

for criteria in criterias:
    logger.info(f"Scores for criteria '{criteria}'")
    gold_answer_scores = [
        instance["score"]["instance"][criteria] for instance in evaluated_gold_answers
    ]
    gold_answer_position_bias = [
        instance["score"]["instance"][f"{criteria}_positional_bias"]
        for instance in evaluated_gold_answers
    ]
    prediction_scores = [
        instance["score"]["instance"][criteria] for instance in evaluated_predictions
    ]
    prediction_scores_position_bias = [
        instance["score"]["instance"][f"{criteria}_positional_bias"]
        for instance in evaluated_predictions
    ]

    logger.info(
        f"Scores of gold answers: {statistics.mean(gold_answer_scores)} +/- {statistics.stdev(gold_answer_scores)}"
    )
    logger.info(
        f"Scores of predicted answers: {statistics.mean(prediction_scores)} +/- {statistics.stdev(prediction_scores)}"
    )
    logger.info(
        f"Positional bias occurrence on gold answers: {statistics.mean(gold_answer_position_bias)}"
    )
    logger.info(
        f"Positional bias occurrence on predicted answers: {statistics.mean(prediction_scores_position_bias)}\n"
    )

"""
Output with 20 examples

Scores for criteria 'answer_relevance'
Scores of gold answers: 0.8875 +/- 0.18978866362906205
Scores of predicted answers: 0.7625 +/- 0.3390679950439998
Positional bias occurrence on gold answers: 0.25
Positional bias occurrence on predicted answers: 0.25

Scores for criteria 'coherence'
Scores of gold answers: 0.8125 +/- 0.2910394257972982
Scores of predicted answers: 0.6875 +/- 0.39632356531129037
Positional bias occurrence on gold answers: 0.3
Positional bias occurrence on predicted answers: 0.3

Scores for criteria 'conciseness'
Scores of gold answers: 1.0 +/- 0.0
Scores of predicted answers: 0.6 +/- 0.5026246899500346
Positional bias occurrence on gold answers: 0
Positional bias occurrence on predicted answers: 0.05
"""
