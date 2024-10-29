from unitxt.blocks import (
    LoadHF,
    Rename,
    Set,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.collections_operators import Wrap
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="yale-nlp/QTSumm"),
    preprocess_steps=[
        Set(
            fields={
                "context_type": "table",
            }
        ),
        Rename(field="table", to_field="context"),
        Rename(field="query", to_field="question"),
        Rename(field="summary", to_field="answers"),
        Wrap(field="answers", inside="list", to_field="answers"),
    ],
    task="tasks.qa.with_context.abstractive[metrics=[metrics.rouge, metrics.bleu, metrics.bert_score.bert_base_uncased, metrics.meteor]]",
    templates=["templates.qa.with_context.qtsumm"],
    __description__="The QTSumm dataset is a large-scale dataset for the task of query-focused summarization over tabular data.",
    __tags__={
        "modality": "table",
        "urls": {"arxiv": "https://arxiv.org/pdf/2305.14303"},
        "languages": ["english"],
    },
)

test_card(card, num_demos=1, demos_pool_size=5, strict=False)
add_to_catalog(card, "cards.qtsumm", overwrite=True)
