import sys

from unitxt import add_to_catalog
from unitxt.blocks import Copy, Rename, Set, TaskCard
from unitxt.loaders import LoadHF
from unitxt.operators import ExecuteExpression, Shuffle

card = TaskCard(
    loader=LoadHF(path="premai-io/birdbench", split="validation"),
    preprocess_steps=[
        Shuffle(page_size=sys.maxsize),
        Rename(
            field_to_field={
                "question_id": "id",
                "question": "utterance",
                "SQL": "query",
                "db_id": "db_id",
                "evidence": "hint",
            }
        ),
        Set(
            fields={
                "dbms": "sqlite",
                "db_type": "local",
                "use_oracle_knowledge": True,
                "num_table_rows_to_add": 0,
                "data": None,
            }
        ),
        ExecuteExpression(
            expression="'bird/'+db_id",
            to_field="db_id",
        ),
        ExecuteExpression(
            expression="str(id)",
            to_field="id",
        ),
        Copy(field="db_id", to_field="db/db_id"),
        Copy(field="db_type", to_field="db/db_type"),
        Copy(field="dbms", to_field="db/dbms"),
        Copy(field="data", to_field="db/data"),
    ],
    task="tasks.text2sql",
    templates="templates.text2sql.all",
)


# test_card(
#     card, num_demos=0, demos_pool_size=0, demos_taken_from="test"
# )

add_to_catalog(
    card,
    "cards.text2sql.bird",
    overwrite=True,
)

# from unitxt import evaluate, load_dataset

# ds = load_dataset("card=cards.text2sql.bird,template_card_index=0")
# scores = evaluate(predictions=ds["validation"]["target"], data=ds["validation"])
