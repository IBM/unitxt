from typing import List, Tuple

from .loaders import Loader, LoadHF
from .operator import (
    InstanceOperator,
    SequentialInstanceOperator,
    SequentialOperator,
    StreamingOperator,
)
from .splitters import SplitRandomMix


def to_non_sequential_operators(operator: StreamingOperator) -> List[StreamingOperator]:
    if not isinstance(operator, SequentialOperator):
        return [operator]
    to_return = []
    for step in operator.steps:
        to_return.extend(to_non_sequential_operators(step))
    return to_return


# flake8: noqa: C901
# separate loading from all rest (for card_profiling business)
# leave loading as is, and for the rest:
# unfold SequentialOperator-s, and replace each maximal subsequence of InstanceOperator-s by SequentialInstanceOperator
def simplify_recipe_steps(
    recipe_steps: List[StreamingOperator]
) -> List[StreamingOperator]:
    def same_streams_next_chunk_step() -> bool:
        if len(next_chunk) == 0:
            return True
        if (next_chunk[0].apply_to_streams is None) != (step.apply_to_streams is None):
            return False
        if (next_chunk[0].dont_apply_to_streams is None) != (
            step.dont_apply_to_streams is None
        ):
            return False
        if next_chunk[0].apply_to_streams is not None:
            if sorted(next_chunk[0].apply_to_streams) != sorted(step.apply_to_streams):
                return False
        if next_chunk[0].dont_apply_to_streams is not None:
            if sorted(next_chunk[0].dont_apply_to_streams) != sorted(
                step.dont_apply_to_streams
            ):
                return False
        return True

    loader_step_index, loader_step = find_step_by_type(recipe_steps, Loader)
    if loader_step_index is None or loader_step_index > 0:
        # no Loader found, or not in expected position, leave this simplification
        return recipe_steps

    # simplify now all the steps:
    rest_steps = []
    # first - unfold
    for step in recipe_steps:  # [1:]:
        rest_steps.extend(to_non_sequential_operators(step))

    # then fold back to sequences of matching instance operators
    to_return = []
    next_chunk = []
    for step in rest_steps:
        if isinstance(step, InstanceOperator) and same_streams_next_chunk_step():
            next_chunk.append(step)
            continue
        if next_chunk:
            if len(next_chunk) > 1:
                to_return.append(
                    SequentialInstanceOperator(
                        steps=next_chunk,
                        apply_to_streams=next_chunk[0].apply_to_streams,
                        dont_apply_to_streams=next_chunk[0].dont_apply_to_streams,
                    )
                )
            else:
                to_return.append(next_chunk[0])
            next_chunk = []
        if isinstance(step, InstanceOperator):
            next_chunk.append(step)
        else:
            to_return.append(step)

    if next_chunk:
        if len(next_chunk) > 1:
            to_return.append(
                SequentialInstanceOperator(
                    steps=next_chunk,
                    apply_to_streams=next_chunk[0].apply_to_streams,
                    dont_apply_to_streams=next_chunk[0].dont_apply_to_streams,
                )
            )
        else:
            to_return.append(next_chunk[0])

    # in to_return, move to front all the splitters, so that they are executed first:
    if (
        isinstance(to_return[0], LoadHF)
        and isinstance(to_return[1], SequentialInstanceOperator)
        and isinstance(to_return[2], SplitRandomMix)
    ):
        to_return = [to_return[0], to_return[2], to_return[1], *to_return[3:]]
    return to_return


def is_or_contains(operator1: StreamingOperator, operator2: StreamingOperator) -> bool:
    if isinstance(operator1, operator2):
        return True
    if not isinstance(operator1, SequentialOperator):
        return False
    simple_steps_of_operator1 = to_non_sequential_operators(operator1)
    for simple_step in simple_steps_of_operator1:
        if isinstance(simple_step, operator2):
            return True
    return False


def find_step_by_type(
    steps: List[StreamingOperator], operator: StreamingOperator
) -> Tuple[int, StreamingOperator]:
    for i, step in enumerate(steps):
        if is_or_contains(step, operator):
            return i, step
    return None, None
