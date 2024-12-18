from unitxt import add_to_catalog
from unitxt.blocks import Copy, Rename, Set, TaskCard
from unitxt.loaders import LoadHF
from unitxt.text2sql.metrics import ExecutionAccuracy  # noqa

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

# card = TaskCard(
#     loader=LoadHF(path="heegyu/bird-sql-mini-dev", split="validation"),
#     preprocess_steps=[
#         Rename(
#             field_to_field={
#                 "question_id": "id",
#                 "question": "utterance",
#                 "SQL": "query",
#             }
#         ),
#         Set(fields={"dbms": "sqlite"}),
#     ],
#     task="tasks.text2sql",
#     templates="templates.text2sql.all",
# )
# add_to_catalog(
#     card,
#     "cards.text2sql.bird_mini_dev",
#     overwrite=True,
# )

card = TaskCard(
    loader=LoadHF(path="premai-io/birdbench", split="validation"),
    preprocess_steps=[
        Rename(
            field_to_field={
                "question_id": "id",
                "question": "utterance",
                "SQL": "query",
                "db_id": "db_id",
            }
        ),
        Set(
            fields={
                "dbms": "sqlite",
                "db_type": "sqlite",
                "use_oracle_knowledge": True,
                "num_table_rows_to_add": 0,
            }
        ),
        Copy(field="db_id", to_field="schema/db_id"),
        Copy(field="db_type", to_field="schema/db_type"),
        Copy(field="num_table_rows_to_add", to_field="schema/num_table_rows_to_add"),
    ],
    task="tasks.text2sql",
    templates="templates.text2sql.all",
)

# test_card(
#     card,
# )

add_to_catalog(
    card,
    "cards.text2sql.bird",
    overwrite=True,
)

# from unitxt import evaluate, load_dataset

# ds = load_dataset("card=cards.text2sql.bird,template_card_index=0,loader_limit=10")
# scores = evaluate(predictions=ds["validation"]["target"], data=ds["validation"])
