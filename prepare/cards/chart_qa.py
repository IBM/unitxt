from unitxt.blocks import LoadHF, Set, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.image_operators import ToImage
from unitxt.operators import Rename
from unitxt.splitters import RenameSplits
from unitxt.collections_operators import Wrap
from unitxt.test_utils.card import test_card
from unitxt.templates import MultiReferenceTemplate


card = TaskCard(
    loader=LoadHF(path="HuggingFaceM4/ChartQA"),
    preprocess_steps=[
        RenameSplits(mapper={"train": "train", "val": "validation", "test": "test"}),
        Rename(field="label", to_field="answers"),
        Rename(field="query", to_field="question"),
        ToImage(field="image", to_field="context"),
        Set(fields={"context_type": "image"}),
    ],
    task="tasks.qa.with_context.abstractive[metrics=[metrics.relaxed_correctness]]",
    templates="templates.qa.with_context.all",
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
        Wrap(field="answer", inside="list", to_field="answers"),
        ToImage(field="image", to_field="context"),
        Set(fields={"context_type": "image"}),
    ],
    task="tasks.qa.with_context.with_type[metrics=[metrics.relaxed_correctness]]",
    templates="templates.qa.with_context.all",
    default_template=MultiReferenceTemplate(
        input_format="{context}\n{question}\nAnswer the question using a single word.",
        references_field="answers",
        __description__="lmms-evals default template for chartqa.",
    ),
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
