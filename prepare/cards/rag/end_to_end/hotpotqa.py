import json

from unitxt import add_to_catalog
from unitxt.blocks import TaskCard
from unitxt.collections_operators import Explode, Wrap
from unitxt.loaders import LoadHF
from unitxt.operators import Copy, Deduplicate, Set, ZipFieldValues
from unitxt.splitters import SplitRandomMix
from unitxt.string_operators import Join, Replace
from unitxt.templates import InputOutputTemplate
from unitxt.test_utils.card import test_card

# Benchmark
benchmark_card = TaskCard(
    loader=LoadHF(
        path="hotpotqa/hotpot_qa",
        name="distractor",
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        SplitRandomMix(
            {
                "test": "train[30%]",
                "train": "train[70%]",
            }),
        Copy(
            field_to_field={
                "question": "question",
                "id": "question_id",
                "level": "metadata_field/level"
            },
        ),
        Copy(
            field="context/title",
            to_field="reference_context_ids",
        ),
        Join(
            field="context/sentences",
            by=" ",
            to_field="reference_contexts",
            process_every_value=True,
        ),
        Set(
            fields={
                "is_answerable_label": True,
            }
        ),
        Wrap(
            field="answer",
            inside="list",
            to_field="reference_answers",
        ),
    ],
    task="tasks.rag.end_to_end",
    templates={"default": "templates.rag.end_to_end.json_predictions"},
    __tags__={"license": "CC BY-SA 4.0"},
    __description__="""HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.
HotpotQA is a new dataset with 113k Wikipedia-based question-answer pairs with four key features: (1) the questions require finding and reasoning over multiple supporting documents to answer; (2) the questions are diverse and not constrained to any pre-existing knowledge bases or knowledge schemas; (3) we provide sentence-level supporting facts required for reasoning, allowingQA systems to reason with strong supervision and explain the predictions; (4) we offer a new type of factoid comparison questions to test QA systems ability to extract relevant facts and perform necessary comparison.
    """,
)
wrong_answer = {
    "contexts": ["hi"],
    "is_answerable": True,
    "answer": "Don't know",
    "context_ids": ["id0"],
}

test_card(
    benchmark_card,
    strict=True,
    full_mismatch_prediction_values=[json.dumps(wrong_answer)],
    debug=False,
    demos_taken_from="test",
    demos_pool_size=5,
)

add_to_catalog(benchmark_card, "cards.rag.benchmark.hotpotqa.en", overwrite=True)


# Documents
documents_card = TaskCard(
    loader=LoadHF(
        path="hotpotqa/hotpot_qa",
        name="distractor",
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        Join(
            field="context/sentences",
            by=" ",
            to_field="context_sentences",
            process_every_value=True,
        ),
        ZipFieldValues(
            fields= ["context/title", "context_sentences"],
            to_field = "documents"
        ),
        Explode(
            field =  "documents",
            to_field = "document"
        ),

        Copy(field="document/0", to_field="document_id"),
        Copy(field="document/0", to_field="title"),
        Replace(field="document/1",old="\xa0", new = " "),
        Wrap(field="document/1", inside="list", to_field="passages"),

        Set(
            fields={
                "metadata_field": {},
            }
        ),
        Deduplicate(by=["document_id"]),
    ],
    task="tasks.rag.corpora",
    templates={
        "empty": InputOutputTemplate(
            input_format="",
            output_format="",
        ),
    },
    __tags__={"license": "CC BY-SA 4.0"},
    __description__="""HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.
HotpotQA is a new dataset with 113k Wikipedia-based question-answer pairs with four key features: (1) the questions require finding and reasoning over multiple supporting documents to answer; (2) the questions are diverse and not constrained to any pre-existing knowledge bases or knowledge schemas; (3) we provide sentence-level supporting facts required for reasoning, allowingQA systems to reason with strong supervision and explain the predictions; (4) we offer a new type of factoid comparison questions to test QA systems ability to extract relevant facts and perform necessary comparison.
""",
)

# Not testing card, because documents are not evaluated.
add_to_catalog(documents_card, "cards.rag.documents.hotpotqa.en", overwrite=True)
