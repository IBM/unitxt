from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF
from unitxt.operators import MapInstanceValues, Rename, Set
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="glue", name="mrpc", streaming=False),
    preprocess_steps=[
        MapInstanceValues(
            mappers={"label": {"0": "not equivalent", "1": "equivalent"}}
        ),
        Rename(field="sentence1", to_field="text_a"),
        Rename(field="sentence2", to_field="text_b"),
        Set(
            fields={
                "classes": ["not equivalent", "equivalent"],
                "text_a_type": "sentence",
                "text_b_type": "sentence",
                "type_of_relation": "semantic equivalence",
            }
        ),
    ],
    task="tasks.classification.multi_class.relation",
    templates="templates.classification.multi_class.relation.all",
    __tags__={
        "annotations_creators": "other",
        "arxiv": "1804.07461",
        "flags": ["coreference-nli", "paraphrase-identification", "qa-nli"],
        "language": "en",
        "language_creators": "other",
        "license": "other",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "10K<n<100K",
        "source_datasets": "original",
        "task_categories": "text-classification",
        "task_ids": [
            "acceptability-classification",
            "natural-language-inference",
            "semantic-similarity-scoring",
            "sentiment-classification",
            "text-scoring",
        ],
    },
    __description__=(
        "The Microsoft Research Paraphrase Corpus (Dolan & Brockett, 2005) is a corpus of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent… See the full description on the dataset page: https://huggingface.co/datasets/nyu-mll/glue."
    ),
)

test_card(card)
add_to_catalog(card, "cards.mrpc", overwrite=True)
