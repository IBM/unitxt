from unitxt.operators import (
    Apply,
    CopyFields,
    DeleteSplits,
    JoinStreams,
    MapInstanceValues,
    RenameFields,
    SelectFields,
)
from unitxt.splitters import RenameSplits

mt_bench_rating_hf_space_processing_steps = [
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
        field_to_field={"choices": "model_output"}, apply_to_streams=["model_answer"]
    ),
    Apply(
        "model_id",
        function=str.lower,
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
        field_to_field={"model_output/0/turns": "model_output"},
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
