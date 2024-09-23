from unitxt import add_to_catalog
from unitxt.operator import SequentialOperator
from unitxt.operators import (
    Apply,
    Copy,
    FilterByCondition,
    Rename,
    SelectFields,
    Set,
)
from unitxt.splitters import RenameSplits
from unitxt.stream_operators import DeleteSplits, JoinStreams

arena_hard_scores = ["A=B", "A>B", "A>>B", "B>A", "B>>A"]

arena_hard_hf_space_processing_steps = SequentialOperator(
    steps=[
        # region Question file
        Rename(field_to_field={"cluster": "group"}, apply_to_streams=["questions"]),
        Copy(
            field_to_field={"turns/0/content": "model_input"},
            apply_to_streams=["questions"],
        ),
        # endregion
        # region Answers file processing
        Copy(
            field_to_field={
                "choices/0/turns/0/content": "model_output",
                "choices/0/turns/0/token_len": "model_output_token_len",
            },
            apply_to_streams=["model_answer"],
        ),
        Apply(
            "model_id",
            function="str.lower",
            to_field="model_id",
            apply_to_streams=["model_answer"],
        ),
        # endregion
        # region Judgment file
        Copy(
            field_to_field={
                "games/0/user_prompt": "judge_input_model_1_ordered_first",
                "games/1/user_prompt": "judge_input_model_2_ordered_first",
                "games/0/judgment": "judge_output_model_1_ordered_first",
                "games/1/judgment": "judge_output_model_2_ordered_first",
                "games/0/score": "score_model_1_ordered_first",
                "games/1/score": "score_model_2_ordered_first",
            },
            apply_to_streams=["judgment"],
        ),
        Rename(
            field_to_field={"model": "model_2", "judge": "judge_model_id"},
            apply_to_streams=["judgment"],
        ),
        Set(fields={"model_1": "gpt-4-0314"}, apply_to_streams=["judgment"]),
        Apply(
            "judge_input_model_1_ordered_first",
            function="str",
            to_field="judge_input_model_1_ordered_first",
            apply_to_streams=["judgment"],
        ),
        Apply(
            "judge_input_model_2_ordered_first",
            function="str",
            to_field="judge_input_model_2_ordered_first",
            apply_to_streams=["judgment"],
        ),
        Apply(
            "model_1",
            function="str.lower",
            to_field="model_1",
            apply_to_streams=["judgment"],
        ),
        Apply(
            "model_2",
            function="str.lower",
            to_field="model_2",
            apply_to_streams=["judgment"],
        ),
        FilterByCondition(
            values={
                "score_model_1_ordered_first": arena_hard_scores,
                "score_model_2_ordered_first": arena_hard_scores,
            },
            condition="in",
            apply_to_streams=["judgment"],
        ),
        # endregion
        # region Join
        JoinStreams(
            left_stream="questions",
            right_stream="judgment",
            how="inner",
            on=["question_id"],
            new_stream_name="merged_stream",
        ),
        Rename(
            field_to_field={"model_id": "model_1", "model_output": "model_1_output"},
            apply_to_streams=["model_answer"],
        ),
        JoinStreams(
            left_stream="merged_stream",
            right_stream="model_answer",
            how="inner",
            on=["question_id", "model_1"],
            new_stream_name="merged_stream",
        ),
        Rename(
            field_to_field={"model_1": "model_2", "model_1_output": "model_2_output"},
            apply_to_streams=["model_answer"],
        ),
        JoinStreams(
            left_stream="merged_stream",
            right_stream="model_answer",
            how="inner",
            on=["question_id", "model_2"],
            new_stream_name="merged_stream",
        ),
        # endregion
        DeleteSplits(splits=["questions", "model_answer", "judgment"]),
        RenameSplits({"merged_stream": "test"}),
        SelectFields(
            fields=[
                "question_id",
                "category",
                "model_input",
                "model_1",
                "model_2",
                "judge_model_id",
                "model_1_output",
                "model_2_output",
                "score_model_1_ordered_first",
                "score_model_2_ordered_first",
                "judge_input_model_1_ordered_first",
                "judge_input_model_2_ordered_first",
                "judge_output_model_1_ordered_first",
                "judge_output_model_2_ordered_first",
            ]
        ),
    ]
)

add_to_catalog(
    arena_hard_hf_space_processing_steps,
    "operators.arena_hard_hf_space_processing_steps",
    overwrite=True,
)
