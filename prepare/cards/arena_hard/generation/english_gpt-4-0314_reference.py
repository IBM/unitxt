from unitxt.blocks import (
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadCSV
from unitxt.operators import (
    Apply,
    Copy,
    Rename,
    SelectFields,
    Set,
)
from unitxt.stream_operators import DeleteSplits, JoinStreams
from unitxt.test_utils.card import test_card

# 'lmsys/arena-hard-browser' (https://huggingface.co/spaces/lmarena-ai/arena-hard-browser) was gone from HF.
# https://paperswithcode.com/dataset/arena-hard points at git repository  https://github.com/lmarena/arena-hard-auto
# from which the dataset is now fetched:

question_git_repo_file_path = "https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/data/arena-hard-v0.1/question.jsonl"
model_answer_git_repo_file_path = "https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/data/arena-hard-v0.1/model_answer/gpt-4-0314.jsonl"

card = TaskCard(
    loader=LoadCSV(
        files={"questions": question_git_repo_file_path, "model_answer": model_answer_git_repo_file_path},
            file_type="json",
            lines=True,
            data_classification_policy=["public"],
        ),
    preprocess_steps=[
        # region Question file
        Rename(field_to_field={"cluster": "group"}, apply_to_streams=["questions"]),
        Copy(
            field_to_field={"prompt": "model_input"},
            apply_to_streams=["questions"],
        ),
        Set(fields={"reference_model": "gpt-4-0314"}, apply_to_streams=["questions"]),
        # endregion
        # region Answers file processing
        Copy(
            field_to_field={
                "messages/1/content/answer": "reference_model_output"
                # "choices/0/turns/0/token_len": "reference_model_output_token_len",
            },
            apply_to_streams=["model_answer"],
        ),
        Rename(
            field_to_field={"model": "reference_model"},
            apply_to_streams=["model_answer"],
        ),
        Apply(
            "reference_model",
            function="str.lower",
            to_field="reference_model",
            apply_to_streams=["model_answer"],
        ),
        # endregion
        # region Join
        JoinStreams(
            left_stream="questions",
            right_stream="model_answer",
            how="inner",
            on=["uid", "reference_model"],
            new_stream_name="test",
        ),
        DeleteSplits(splits=["questions", "model_answer"]),
        SelectFields(
            fields=[
                "uid",
                "category",
                "model_input",
                "reference_model",
                "reference_model_output",
            ]
        ),
        Rename(
            field_to_field={
                "uid": "question_id",
                "model_input": "input",
                "category": "group",
                "reference_model_output": "output",
            }
        ),
        Set(
            fields={
                "type_of_input": "prompt",
                "type_of_output": "answer",
            }
        ),
    ],
    task="tasks.generation",
    templates=["templates.generation.empty"],
)

test_card(card, demos_taken_from="test", strict=False, loader_limit=100)
add_to_catalog(
    card,
    "cards.arena_hard.generation.english_gpt_4_0314_reference",
    overwrite=True,
)
