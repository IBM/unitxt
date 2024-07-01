import sys

from unitxt import add_to_catalog
from unitxt.blocks import LoadHF, TaskCard
from unitxt.operators import (
    Copy,
    GetItemByIndex,
    RenameFields,
    Set,
    Shuffle,
)
from unitxt.span_lableing_operators import IobExtractor
from unitxt.test_utils.card import test_card

debug = True
if debug:
    from unitxt import load_dataset

    ds = load_dataset(
        "card=cards.universal_ner.en.ewt,metrics=[metrics.ner[zero_division=1.0]],demos_pool_size=10000,num_demos=5,format=formats.llama3_chat,template=templates.span_labeling.extraction.title,system_prompt=system_prompts.empty,"
        "train_refiner=operators.balancers.ner.zero_vs_many_entities[segments_boundaries=[0,1,2]]"
        ",demos_taken_from=train,augmentor=augmentors.no_augmentation,demos_removed_from_data=True,max_test_instances=50"
    )

    ds["test"]["source"][0]


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
            Shuffle(page_size=sys.maxsize),
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
                    "classes": ["Person", "Organization", "Location"],
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
