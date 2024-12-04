import statistics

import numpy as np
from unitxt import get_logger
from unitxt.api import evaluate, load_dataset

logger = get_logger()

# This example demonstrates how to evaluate the quality of LLM as judge
# on a task by using the gold references of a dataset.

# It checks two llama3 based judges - one based on a 8b model and one on a 70b model on a
# summarization dataset.
#
# The results indicate that the 8b model gives a higher score to a wrong prediction over the correct
# prediction in 20% of the examples, and gives a truncated corrected prediction a higher score than
# the correct prediction in 35% of examples.  This means it is not so good as a judge for this task.
#
# On the other hand the 70b model is better.  It always gives a higher score for the correct prediction,
# and in only 5% of the cases it gives the truncated prediction a higher score.
# Note that even the 70b model gives relatively low average score for correct predictions (0.395 +/ 0.17)

# List of metrics to evaluate
metrics_to_check = [
    "metrics.llm_as_judge.rating.llama_3_8b_instruct_ibm_genai_template_mt_bench_single_turn",
    "metrics.llm_as_judge.rating.llama_3_70b_instruct_ibm_genai_template_generic_single_turn",
]

for metric_to_check in metrics_to_check:
    # The dataset used to evaluate the metrics based on its gold answers
    dataset = load_dataset(
        card="cards.xsum",
        template="templates.summarization.abstractive.formal",
        metrics=[metric_to_check],
        loader_limit=20,
    )
    test_dataset = dataset["test"]

    # Prepare three sets of predictions :
    # 1. the correct predictions taken from the gold answer
    # 2. wrong predictions (where the prediction is the gold answer of another question)
    # 3. truncated predictions taken as the first half of the gold answer
    correct_predictions = test_dataset["target"]
    wrong_predictions = [correct_predictions[-1]]
    wrong_predictions.extend(correct_predictions[0:-1])
    truncated_predictions = [
        prediction[: len(prediction) // 2] for prediction in correct_predictions
    ]

    # Evaluate over the correct, wrong and truncated predictions using the defined metric.
    correct_evaluated_dataset = evaluate(
        predictions=correct_predictions, data=test_dataset
    )
    wrong_evaluated_dataset = evaluate(predictions=wrong_predictions, data=test_dataset)
    truncated_evaluated_dataset = evaluate(
        predictions=truncated_predictions, data=test_dataset
    )

    correct_prediction_scores = [
        correct_evaluated_dataset[i]["score"]["instance"]["score"]
        for i in range(len(correct_predictions))
    ]
    wrong_prediction_scores = [
        wrong_evaluated_dataset[i]["score"]["instance"]["score"]
        for i in range(len(wrong_predictions))
    ]
    truncated_prediction_scores = [
        truncated_evaluated_dataset[i]["score"]["instance"]["score"]
        for i in range(len(truncated_predictions))
    ]

    # Print the scores of the metric on each type of prediction.
    # The score of correct predictions should be close to 1 with low standard deviation
    # The score of wrong predictions should be close to 0 with low standard deviation
    # The score of the truncated prediction, should be between the values.
    # Also prints the percent of examples the wrong / truncated prediction get a higher score than the correct prediction.

    logger.info(f"Meta evaluation of metric: {metric_to_check}")
    logger.info(f"Scores of correct predictions: {correct_prediction_scores}")
    logger.info(f"Scores of wrong predictions: {wrong_prediction_scores}")
    logger.info(f"Scores of truncated predictions: {truncated_prediction_scores}")
    logger.info(
        f"Average score of correct predictions: {statistics.mean(correct_prediction_scores)} +/- {statistics.stdev(correct_prediction_scores)}"
    )
    logger.info(
        f"Average score of wrong predictions: {statistics.mean(wrong_prediction_scores)} +/- {statistics.stdev(wrong_prediction_scores)}"
    )
    logger.info(
        f"% Wrong predictions scores greater than correct prediction scores: {np.sum(np.greater(wrong_prediction_scores, correct_prediction_scores)) * 100/ len(correct_predictions)}"
    )
    logger.info(
        f"Average score of truncated predictions: {statistics.mean(truncated_prediction_scores)} +/- {statistics.stdev(truncated_prediction_scores)}"
    )
    logger.info(
        f"% Truncated predictions scores greater than correct prediction scores: {np.sum(np.greater(truncated_prediction_scores, correct_prediction_scores)) * 100/ len(correct_predictions)}"
    )
