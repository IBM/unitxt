from unitxt import get_from_catalog
from unitxt.blocks import LoadHF, Set, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.collections_operators import Wrap
from unitxt.image_operators import ToImage
from unitxt.operators import Rename
from unitxt.splitters import RenameSplits, SplitRandomMix
from unitxt.templates import MultiReferenceTemplate
from unitxt.test_utils.card import test_card
from unitxt.templates import MultiReferenceTemplate
from unitxt.splitters import RenameSplits


templates = get_from_catalog("templates.qa.with_context.all")
template = MultiReferenceTemplate(
    input_format="{context}\n{question}\nAnswer the question using a single word.",
    references_field="answers",
    __description__="lmms-evals default template for chartqa.",
)


card = TaskCard(
    loader=LoadHF(path="vidore/infovqa_train"),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "train[90%]", "validation": "train[5%]", "test": "train[5%]"}
        ),
        Wrap(field="answer", inside="list", to_field="answers"),
        Rename(field="query", to_field="question"),
        ToImage(field="image", to_field="context"),
        Set(fields={"context_type": "image"}),
    ],
    task="tasks.qa.with_context.abstractive[metrics=[metrics.anls]]",
    templates=[template, *templates.items],
    default_template=template,
    __tags__={
        "license": "Unknown",
        "multilinguality": "monolingual",
        "modalities": ["image", "text"],
        "size_categories": "10K<n<100K",
        "task_categories": "question-answering",
        "task_ids": "extractive-qa",
    },
    __description__=(
        "InfographicVQA is a dataset that comprises a diverse collection of infographics along with natural language questions and answers annotations. The collected questions require methods to jointly reason over the document layout, textual content, graphical elements, and data visualizations. We curate the dataset with emphasis on questions that require elementary reasoning and basic arithmetic skills."
    ),
)

test_card(card)
add_to_catalog(card, "cards.info_vqa", overwrite=True)


card = TaskCard(
    loader=LoadHF(
        path="lmms-lab/DocVQA",
        name="InfographicVQA",
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        RenameSplits(mapper={"validation": "test"}),
        ToImage(field="image", to_field="context"),
        Set(fields={"context_type": "image"}),
    ],
    task="tasks.qa.with_context.abstractive[metrics=[metrics.anls]]",
    templates="templates.qa.with_context.all",
    default_template=MultiReferenceTemplate(
        input_format="{context}\n{question}\nAnswer the question using a single word or phrase.",
        references_field="answers",
        __description__="lmms-evals default template for infovqa.",
    ),
    __tags__={
        "license": "apache-2.0",
        "multilinguality": "monolingual",
        "modalities": ["image", "text"],
        "size_categories": "10K<n<100K",
        "task_categories": "question-answering",
        "task_ids": "extractive-qa",
    },
    __description__=(
        "InfographicVQA is a dataset that comprises a diverse collection of infographics along with natural language questions and answers annotations. The collected questions require methods to jointly reason over the document layout, textual content, graphical elements, and data visualizations. We curate the dataset with emphasis on questions that require elementary reasoning and basic arithmetic skills."
    ),
)

test_card(card)
add_to_catalog(card, "cards.info_vqa_lmms_eval", overwrite=True)
