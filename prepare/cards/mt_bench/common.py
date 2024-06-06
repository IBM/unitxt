from unitxt import add_to_catalog
from unitxt.operators import (
    Apply,
    CopyFields,
    FilterByConditionBasedOnFields,
    MapInstanceValues,
    RenameFields,
    SelectFields,
    SequentialOperator,
)
from unitxt.splitters import RenameSplits
from unitxt.stream_operators import DeleteSplits, JoinStreams

mt_bench_rating_hf_space_processing_steps = SequentialOperator(
    steps=[
        RenameFields(
            field_to_field={"turns": "model_input"}, apply_to_streams=["questions"]
        ),
        RenameFields(
            field_to_field={
                "model": "model_id",
                "judge": "judge_model_id",
                "user_prompt": "judge_input",
                "judgment": "judge_output",
            },
            apply_to_streams=["judgment"],
        ),
        RenameFields(
            field_to_field={"choices": "model_output"},
            apply_to_streams=["model_answer"],
        ),
        Apply(
            "model_id",
            function="str.lower",
            to_field="model_id",
            apply_to_streams=["judgment", "model_answer"],
        ),
        MapInstanceValues(
            mappers={
                "model_id": {
                    "vicuna-13b-hao-0515": "vicuna-13b-v1.3",
                    "vicuna-30b-gpt4": "vicuna-33b-v1.3",
                }
            },
            strict=False,
            apply_to_streams=["judgment", "model_answer"],
        ),
        CopyFields(
            field="model_output/0/turns",
            to_field="model_output",
            apply_to_streams=["model_answer"],
        ),
        JoinStreams(
            left_stream="questions",
            right_stream="judgment",
            how="inner",
            on=["question_id"],
            new_stream_name="merged_stream",
        ),
        JoinStreams(
            left_stream="merged_stream",
            right_stream="model_answer",
            how="inner",
            on=["question_id", "model_id"],
            new_stream_name="merged_stream",
        ),
        DeleteSplits(splits=["questions", "model_answer", "judgment"]),
        RenameSplits({"merged_stream": "test"}),
        SelectFields(
            fields=[
                "question_id",
                "category",
                "model_input",
                "reference",
                "turn",
                "model_id",
                "judge_model_id",
                "score",
                "model_output",
                "judge_input",
                "judge_output",
            ]
        ),
    ]
)

mt_bench_pairwise_hf_space_processing_steps = SequentialOperator(
    steps=[
        # Question file
        RenameFields(
            field_to_field={"turns": "model_input"}, apply_to_streams=["questions"]
        ),
        # region Judgment file
        RenameFields(
            field_to_field={
                "judge": "judge_model_id",
                "g1_user_prompt": "judge_input_model_1_ordered_first",
                "g2_user_prompt": "judge_input_model_2_ordered_first",
                "g1_judgment": "judge_output_model_1_ordered_first",
                "g2_judgment": "judge_output_model_2_ordered_first",
                "g1_winner": "winner_model_1_ordered_first",
                "g2_winner": "winner_model_2_ordered_first",
            },
            apply_to_streams=["judgment"],
        ),
        Apply(
            "model_1",
            function="str.lower",
            to_field="model_1",
            apply_to_streams=["judgment"],
        ),
        MapInstanceValues(
            mappers={
                "model_1": {
                    "vicuna-13b-hao-0515": "vicuna-13b-v1.3",
                    "vicuna-30b-gpt4": "vicuna-33b-v1.3",
                }
            },
            strict=False,
            apply_to_streams=["judgment"],
        ),
        Apply(
            "model_2",
            function="str.lower",
            to_field="model_2",
            apply_to_streams=["judgment"],
        ),
        MapInstanceValues(
            mappers={
                "model_2": {
                    "vicuna-13b-hao-0515": "vicuna-13b-v1.3",
                    "vicuna-30b-gpt4": "vicuna-33b-v1.3",
                }
            },
            strict=False,
            apply_to_streams=["judgment"],
        ),
        CopyFields(
            field="judge_model_id/0",
            to_field="judge_model_id",
            apply_to_streams=["judgment"],
        ),
        FilterByConditionBasedOnFields(
            values={"winner_model_1_ordered_first": "winner_model_2_ordered_first"},
            condition="eq",
            apply_to_streams=["judgment"],
        ),
        CopyFields(
            field_to_field={"winner_model_1_ordered_first": "winner"},
            apply_to_streams=["judgment"],
        ),
        # endregion
        # region Answers file processing
        RenameFields(
            field_to_field={"choices": "model_output"},
            apply_to_streams=["model_answer"],
        ),
        Apply(
            "model_id",
            function="str.lower",
            to_field="model_id",
            apply_to_streams=["model_answer"],
        ),
        MapInstanceValues(
            mappers={
                "model_id": {
                    "vicuna-13b-hao-0515": "vicuna-13b-v1.3",
                    "vicuna-30b-gpt4": "vicuna-33b-v1.3",
                }
            },
            strict=False,
            apply_to_streams=["model_answer"],
        ),
        CopyFields(
            field="model_output/0/turns",
            to_field="model_output",
            apply_to_streams=["model_answer"],
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
        RenameFields(
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
        RenameFields(
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
                "reference",
                "model_1",
                "model_2",
                "turn",
                "judge_model_id",
                "model_1_output",
                "model_2_output",
                "winner_model_1_ordered_first",
                "winner_model_2_ordered_first",
                "winner",
                "judge_input_model_1_ordered_first",
                "judge_input_model_2_ordered_first",
                "judge_output_model_1_ordered_first",
                "judge_output_model_2_ordered_first",
            ]
        ),
    ]
)

add_to_catalog(
    mt_bench_rating_hf_space_processing_steps,
    "operators.mt_bench.rating_hf_space_processing_steps",
    overwrite=True,
)
add_to_catalog(
    mt_bench_pairwise_hf_space_processing_steps,
    "operators.mt_bench.pairwise_hf_space_processing_steps",
    overwrite=True,
)
