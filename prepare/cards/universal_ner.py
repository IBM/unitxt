import sys

from unitxt import add_to_catalog
from unitxt.blocks import TaskCard
from unitxt.loaders import LoadIOB
from unitxt.operators import (
    Copy,
    Rename,
    Set,
    Shuffle,
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

# URL mapping for IOB files based on Universal NER GitHub structure
UNER_PREFIX = "https://raw.githubusercontent.com/UniversalNER/"
UNER_DATASETS = {
    "ceb_gja": {
        "test": "UNER_Cebuano-GJA/master/ceb_gja-ud-test.iob2",
    },
    "zh_gsd": {
        "train": "UNER_Chinese-GSD/master/zh_gsd-ud-train.iob2",
        "dev": "UNER_Chinese-GSD/master/zh_gsd-ud-dev.iob2",
        "test": "UNER_Chinese-GSD/master/zh_gsd-ud-test.iob2",
    },
    "zh_gsdsimp": {
        "train": "UNER_Chinese-GSDSIMP/master/zh_gsdsimp-ud-train.iob2",
        "dev": "UNER_Chinese-GSDSIMP/master/zh_gsdsimp-ud-dev.iob2",
        "test": "UNER_Chinese-GSDSIMP/master/zh_gsdsimp-ud-test.iob2",
    },
    "zh_pud": {
        "test": "UNER_Chinese-PUD/master/zh_pud-ud-test.iob2",
    },
    "hr_set": {
        "train": "UNER_Croatian-SET/main/hr_set-ud-train.iob2",
        "dev": "UNER_Croatian-SET/main/hr_set-ud-dev.iob2",
        "test": "UNER_Croatian-SET/main/hr_set-ud-test.iob2",
    },
    "da_ddt": {
        "train": "UNER_Danish-DDT/main/da_ddt-ud-train.iob2",
        "dev": "UNER_Danish-DDT/main/da_ddt-ud-dev.iob2",
        "test": "UNER_Danish-DDT/main/da_ddt-ud-test.iob2",
    },
    "en_ewt": {
        "train": "UNER_English-EWT/master/en_ewt-ud-train.iob2",
        "dev": "UNER_English-EWT/master/en_ewt-ud-dev.iob2",
        "test": "UNER_English-EWT/master/en_ewt-ud-test.iob2",
    },
    "en_pud": {
        "test": "UNER_English-PUD/master/en_pud-ud-test.iob2",
    },
    "de_pud": {
        "test": "UNER_German-PUD/master/de_pud-ud-test.iob2",
    },
    "pt_bosque": {
        "train": "UNER_Portuguese-Bosque/master/pt_bosque-ud-train.iob2",
        "dev": "UNER_Portuguese-Bosque/master/pt_bosque-ud-dev.iob2",
        "test": "UNER_Portuguese-Bosque/master/pt_bosque-ud-test.iob2",
    },
    "pt_pud": {
        "test": "UNER_Portuguese-PUD/master/pt_pud-ud-test.iob2",
    },
    "ru_pud": {
        "test": "UNER_Russian-PUD/master/ru_pud-ud-test.iob2",
    },
    "sr_set": {
        "train": "UNER_Serbian-SET/main/sr_set-ud-train.iob2",
        "dev": "UNER_Serbian-SET/main/sr_set-ud-dev.iob2",
        "test": "UNER_Serbian-SET/main/sr_set-ud-test.iob2",
    },
    "sk_snk": {
        "train": "UNER_Slovak-SNK/master/sk_snk-ud-train.iob2",
        "dev": "UNER_Slovak-SNK/master/sk_snk-ud-dev.iob2",
        "test": "UNER_Slovak-SNK/master/sk_snk-ud-test.iob2",
    },
    "sv_pud": {
        "test": "UNER_Swedish-PUD/master/sv_pud-ud-test.iob2",
    },
    "sv_talbanken": {
        "train": "UNER_Swedish-Talbanken/master/sv_talbanken-ud-train.iob2",
        "dev": "UNER_Swedish-Talbanken/master/sv_talbanken-ud-dev.iob2",
        "test": "UNER_Swedish-Talbanken/master/sv_talbanken-ud-test.iob2",
    },
    "tl_trg": {
        "test": "UNER_Tagalog-TRG/master/tl_trg-ud-test.iob2",
    },
    "tl_ugnayan": {
        "test": "UNER_Tagalog-Ugnayan/master/tl_ugnayan-ud-test.iob2",
    },
}

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
    # Build file URLs for this sub_task
    files = {}
    for split, file_path in UNER_DATASETS[sub_task].items():
        files[split] = UNER_PREFIX + file_path

    card = TaskCard(
        loader=LoadIOB(
            files=files,
            data_classification_policy=["public"],
        ),
        preprocess_steps=[
            # The dataset is sorted by classes
            Shuffle(page_size=sys.maxsize),
            Rename(
                field_to_field={"ner_tags": "labels"},
            ),
            # IOB loader already provides string labels, no need to map indices
            IobExtractor(
                labels=["Person", "Organization", "Location"],
                begin_labels=["B-PER", "B-ORG", "B-LOC"],
                inside_labels=["I-PER", "I-ORG", "I-LOC"],
                outside_label="O",
            ),
            Copy(
                field_to_field={
                    "spans/*/start": "spans_starts",
                    "spans/*/end": "spans_ends",
                    "spans/*/label": "labels",
                },
                get_default=[],
                not_exist_ok=True,
            ),
            Set(
                fields={
                    "entity_types": ["Person", "Organization", "Location"],
                }
            ),
        ],
        task="tasks.span_labeling.extraction",
        templates="templates.span_labeling.extraction.all",
        __tags__={
            "arxiv": "2311.09122",
            "language": [
                "ceb",
                "da",
                "de",
                "en",
                "hr",
                "pt",
                "ru",
                "sk",
                "sr",
                "sv",
                "tl",
                "zh",
            ],
            "license": "cc-by-sa-4.0",
            "region": "us",
            "task_categories": "token-classification",
        },
        __description__=(
            "Universal Named Entity Recognition (UNER) aims to fill a gap in multilingual NLP: high quality NER datasets in many languages with a shared tagset. UNER is modeled after the Universal Dependencies project, in that it is intended to be a large community annotation effort with language-universal guidelines. Further, we use the same text corpora as Universal Dependencies… See the full description on the dataset page: https://huggingface.co/datasets/universalner/universal_ner"
        ),
    )

    if sub_task == "en_ewt":
        test_card(card)

    sub_task = sub_task.replace("_", ".")

    add_to_catalog(card, f"cards.universal_ner.{sub_task}", overwrite=True)
