from unitxt import get_from_catalog
from unitxt.blocks import LoadHF, Set, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.collections_operators import Wrap
from unitxt.image_operators import ToImage
from unitxt.operators import Rename, Shuffle
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

templates = get_from_catalog("templates.qa.with_context.all")
default_template = get_from_catalog("templates.qa.with_context.chart_qa")

card = TaskCard(
    loader=LoadHF(path="HuggingFaceM4/ChartQA"),
    preprocess_steps=[
        Shuffle(),
        RenameSplits(mapper={"train": "train", "val": "validation", "test": "test"}),
        Rename(field="label", to_field="answers"),
        Rename(field="query", to_field="question"),
        ToImage(field="image", to_field="context"),
        Set(fields={"context_type": "image"}),
    ],
    task="tasks.qa.with_context",
    templates=[default_template, *templates.items],
    __tags__={
        "license": "GPL-3.0",
        "multilinguality": "monolingual",
        "modalities": ["image", "text"],
        "size_categories": "10K<n<100K",
        "task_categories": "question-answering",
        "task_ids": "extractive-qa",
    },
    __description__=(
        "ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning."
    ),
)

test_card(card)
add_to_catalog(card, "cards.chart_qa", overwrite=True)


card = TaskCard(
    loader=LoadHF(path="lmms-lab/ChartQA"),
    preprocess_steps=[
        Shuffle(),
        Wrap(field="answer", inside="list", to_field="answers"),
        ToImage(field="image", to_field="context"),
        Set(fields={"context_type": "image"}),
    ],
    task="tasks.qa.with_context.with_type[metrics=[metrics.relaxed_correctness]]",
    templates=[default_template, *templates.items],
    __tags__={
        "license": "GPL-3.0",
        "multilinguality": "monolingual",
        "modalities": ["image", "text"],
        "size_categories": "10K<n<100K",
        "task_categories": "question-answering",
        "task_ids": "extractive-qa",
    },
    __description__=(
        "ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning."
    ),
)

test_card(card)
add_to_catalog(card, "cards.chart_qa_lmms_eval", overwrite=True)
