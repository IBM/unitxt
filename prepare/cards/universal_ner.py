from unitxt import add_to_catalog
from unitxt.blocks import LoadHF, TaskCard
from unitxt.operators import (
    AddFields,
    CopyFields,
    GetItemByIndex,
    RenameFields,
)
from unitxt.span_lableing_operators import IobExtractor
from unitxt.test_utils.card import test_card

sub_tasks = [
    "ceb_gja",
    "zh_gsd",
    "zh_gsdsimp",
    "zh_pud",
    "hr_set",
    "da_ddt",
    "en_ewt",
    "en_pud",
    "de_pud",
    "pt_bosque",
    "pt_pud",
    "ru_pud",
    "sr_set",
    "sk_snk",
    "sv_pud",
    "sv_talbanken",
    "tl_trg",
    "tl_ugnayan",
]

classes = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
]

for sub_task in sub_tasks:
    card = TaskCard(
        loader=LoadHF(
            path="universalner/universal_ner",
            name=sub_task,
            requirements_list=["conllu"],
        ),
        preprocess_steps=[
            RenameFields(
                field_to_field={"ner_tags": "labels"},
            ),
            GetItemByIndex(
                field="labels", items_list=classes, process_every_value=True
            ),
            IobExtractor(
                labels=["Person", "Organization", "Location"],
                begin_labels=["B-PER", "B-ORG", "B-LOC"],
                inside_labels=["I-PER", "I-ORG", "I-LOC"],
                outside_label="O",
            ),
            CopyFields(
                field_to_field={
                    "spans/*/start": "spans_starts",
                    "spans/*/end": "spans_ends",
                    "spans/*/label": "labels",
                },
                get_default=[],
                not_exist_ok=True,
            ),
            AddFields(
                fields={
                    "text_type": "text",
                    "class_type": "entity type",
                    "classes": ["Person", "Organization", "Location"],
                }
            ),
        ],
        task="tasks.span_labeling.extraction",
        templates="templates.span_labeling.extraction.all",
        __tags__={
            "dataset_info_tags": [
                "task_categories:token-classification",
                "language:ceb",
                "language:da",
                "language:de",
                "language:en",
                "language:hr",
                "language:pt",
                "language:ru",
                "language:sk",
                "language:sr",
                "language:sv",
                "language:tl",
                "language:zh",
                "license:cc-by-sa-4.0",
                "arxiv:2311.09122",
                "region:us",
            ]
        },
    )

    if sub_task == "en_ewt":
        test_card(card)

    sub_task = sub_task.replace("_", ".")

    add_to_catalog(card, f"cards.universal_ner.{sub_task}", overwrite=True)
