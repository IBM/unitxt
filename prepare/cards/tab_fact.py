from unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    RenameFields,
    SerializeTableAsIndexedRowMajor,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

# Set unitxt.settings.allow_unverified_code=True or environment variable: UNITXT_ALLOW_UNVERIFIED_CODE to True

card = TaskCard(
    loader=LoadHF(path="ibm/tab_fact", streaming=False),
    preprocess_steps=[
        "splitters.small_no_test",
        SerializeTableAsIndexedRowMajor(field_to_field=[["table", "table_serialized"]]),
        RenameFields(
            field_to_field={"table_serialized": "text_a", "statement": "text_b"}
        ),
        MapInstanceValues(mappers={"label": {"0": "refuted", "1": "entailed"}}),
        AddFields(
            fields={
                "type_of_relation": "entailment",
                "text_a_type": "Table",
                "text_b_type": "Statement",
                "classes": ["refuted", "entailed"],
            }
        ),
    ],
    task="tasks.classification.multi_class.relation",
    templates="templates.classification.multi_class.relation.all",
    __tags__={
        "arxiv": "1909.02164",
        "license": "cc-by-4.0",
        "region": "us",
        "task_categories": "text-classification",
    },
    __description__=(
        "The problem of verifying whether a textual hypothesis holds the truth based on the given evidence, also known as fact verification, plays an important role in the study of natural language understanding and semantic representation. However, existing studies are restricted to dealing with unstructured textual evidence (e.g., sentences and passages, a pool of passages), while verification using structured forms of evidence, such as tables, graphs, and databases, remains unexplored. TABFACT is large scale dataset with 16k Wikipedia tables as evidence for 118k human annotated statements designed for fact verification with semi-structured evidence. The statements are labeled as either ENTAILED or REFUTED. TABFACT is challenging since it involves both soft linguistic reasoning and hard symbolic reasoning."
    ),
)

test_card(card)
add_to_catalog(card, "cards.tab_fact", overwrite=True)
