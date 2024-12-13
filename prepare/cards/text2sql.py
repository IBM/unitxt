from unitxt import add_to_catalog
from unitxt.blocks import Rename, Set, TaskCard
from unitxt.loaders import LoadHF
from unitxt.text2sql.metrics import ExecutionAccuracy  # noqa
from unitxt.text2sql.templates import Text2SQLInputOutputTemplate  # noqa

# add_to_catalog(
#     Task(
#         input_fields={"id": str, "utterance": str, "db_id": str, "dbms": str},
#         reference_fields={"query": str},
#         prediction_type=str,
#         metrics=["metrics.text2sql.execution_accuracy"],
#     ),
#     "tasks.text2sql",
# )


# import pandas as pd
# df = pd.read_json("hf://datasets/heegyu/bird-sql-mini-dev/validation.json")
# print(df.columns.tolist())
# ['question_id', 'db_id', 'question', 'evidence', 'SQL', 'difficulty']

card = TaskCard(
    loader=LoadHF(path="heegyu/bird-sql-mini-dev", split="validation"),
    preprocess_steps=[
        Rename(
            field_to_field={
                "question_id": "id",
                "question": "utterance",
                "SQL": "query",
            }
        ),
        Set(fields={"dbms": "sqlite"}),
    ],
    task="tasks.text2sql",
    templates="templates.text2sql.all",
)
add_to_catalog(
    card,
    "cards.text2sql.bird_mini_dev",
    overwrite=True,
)

card = TaskCard(
    loader=LoadHF(path="premai-io/birdbench", split="validation"),
    preprocess_steps=[
        Rename(
            field_to_field={
                "question_id": "id",
                "question": "utterance",
                "SQL": "query",
            }
        ),
        Set(fields={"dbms": "sqlite"}),
    ],
    task="tasks.text2sql",
    templates="templates.text2sql.all",
)
add_to_catalog(
    card,
    "cards.text2sql.bird",
    overwrite=True,
)
