from unitxt.blocks import (
    LoadHF,
    MapInstanceValues,
    Rename,
    Set,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.splitters import SplitRandomMix
from unitxt.struct_data_operators import GetMaskedTableHeader
from unitxt.test_utils.card import test_card

# Set unitxt.settings.allow_unverified_code=True or environment variable: UNITXT_ALLOW_UNVERIFIED_CODE to True

card = TaskCard(
    loader=LoadHF(
        path="ibm/tab_fact",
        streaming=False,
        data_classification_policy=["public", "proprietary"],
    ),
    preprocess_steps=[
        SplitRandomMix(
            mix={
                "train": "train[50%]",
                "validation": "train[50%]",
                "test": "test+validation",
            }
        ),
        Rename(field_to_field={"table": "text_a", "statement": "text_b"}),
        MapInstanceValues(mappers={"label": {"0": "refuted", "1": "entailed"}}),
        Set(
            fields={
                "type_of_relation": "entailment",
                "text_a_type": "Table",
                "text_b_type": "Statement",
                "classes": ["refuted", "entailed"],
            }
        ),
        GetMaskedTableHeader(field="text_a"),
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
        "The problem of verifying whether a textual hypothesis holds the truth based on the given evidence, also known as fact verification, plays an important role in the study of natural language understanding and semantic representation. However, existing studies are restricted to dealing with unstructured textual evidence (e.g., sentences and passages, a pool of passages), while verification using structured forms of evidence, such as tables, graphs, and databases, remains unexplored. TABFACT is large scale dataset with 16k Wikipedia tables as evidence for 118k human annotated statements designed for fact verification with semi-structured evidence… See the full description on the dataset page: https://huggingface.co/datasets/ibm/tab_fact"
    ),
)

test_card(card)
add_to_catalog(card, "cards.tab_fact__masked_header", overwrite=True)
