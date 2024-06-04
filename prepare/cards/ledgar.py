from datasets import load_dataset_builder
from unitxt import add_to_catalog
from unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    TaskCard,
)
from unitxt.test_utils.card import test_card

dataset_name = "ledgar"

ds_builder = load_dataset_builder("lex_glue", dataset_name)
classlabels = ds_builder.info.features["label"]

mappers = {}
for i in range(len(classlabels.names)):
    mappers[str(i)] = classlabels.names[i]

card = TaskCard(
    loader=LoadHF(path="lex_glue", name=f"{dataset_name}"),
    preprocess_steps=[
        MapInstanceValues({"label": mappers}),
        AddFields(
            fields={
                "classes": classlabels.names,
                "text_type": "text",
                "type_of_class": "contractual clauses",
            }
        ),
    ],
    task="tasks.classification.multi_class",
    templates="templates.classification.multi_class.all",
    __tags__={
        "annotations_creators": "found",
        "arxiv": ["2110.00976", "2109.00904", "1805.01217", "2104.08671"],
        "flags": ["croissant"],
        "language": "en",
        "language_creators": "found",
        "license": "cc-by-4.0",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "10K<n<100K",
        "source_datasets": "extended",
        "task_categories": ["question-answering", "text-classification"],
        "task_ids": [
            "multi-class-classification",
            "multi-label-classification",
            "multiple-choice-qa",
            "topic-classification",
        ],
    },
    __description__=(
        'Dataset Card for "LexGLUE" Dataset Summary Inspired by the recent widespread use of the GLUE multi-task benchmark NLP dataset (Wang et al., 2018), the subsequent more difficult SuperGLUE (Wang et al., 2019), other previous multi-task NLP benchmarks (Conneau and Kiela, 2018; McCann et al., 2018), and similar initiatives in other domains (Peng et al., 2019), we introduce the Legal General Language Understanding Evaluation (LexGLUE) benchmark, a benchmark dataset to… See the full description on the dataset page: https://huggingface.co/datasets/coastalcph/lex_glue.\n'
    ),
)
test_card(card, debug=False)
add_to_catalog(card, f"cards.{dataset_name}", overwrite=True)
