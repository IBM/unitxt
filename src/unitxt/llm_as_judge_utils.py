from typing import Dict

from .llm_as_judge_constants import (
    EVALUATORS_METADATA,
    EvaluatorMetadata,
    EvaluatorNameEnum,
)


def get_parsed_context(context: Dict[str, str]):
    return (
        "\n".join([f"{key}: {value}" for key, value in context.items()])
        if len(context) > 1
        or not (len(context) == 1 and next(iter(context.keys())).lower() == "context")
        else context[next(iter(context.keys()))]
    )


def get_evaluator_metadata(
    name: EvaluatorNameEnum,
) -> EvaluatorMetadata:  # , evaluator_type: EvaluatorTypeEnum) -> EvaluatorMetadata:
    evaluator_search = [
        e for e in EVALUATORS_METADATA if e.name == name
    ]  # and e.evaluator_type == evaluator_type]
    if len(evaluator_search) == 0:
        # raise ValueError(f'A {evaluator_type} evaluator with id {name} does not exist.')
        raise ValueError(f"An evaluator with id {name} does not exist.")
    if len(evaluator_search) > 1:
        # raise ValueError(f'A {evaluator_type} evaluator with id {name} matched several models.')
        raise ValueError(f"An evaluator with id {name} matched several models.")
    return evaluator_search[0]


def rank_indexes(numbers):
    # Generate the initial list of indices
    indices = list(range(len(numbers)))

    # Sort the indices based on the corresponding values in numbers (descending order)
    sorted_indices = sorted(indices, key=lambda x: -numbers[x])

    # Initialize a list to hold the rankings
    rankings = [0] * len(numbers)

    # Assign rankings
    current_rank = 0
    for i in range(len(sorted_indices)):
        if i > 0 and numbers[sorted_indices[i]] != numbers[sorted_indices[i - 1]]:
            current_rank = i
        rankings[sorted_indices[i]] = current_rank

    return rankings
