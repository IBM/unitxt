from unitxt.blocks import (
    LoadHF,
    MapInstanceValues,
    Rename,
    Set,
    SplitRandomMix,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="glue", name="wnli", data_classification_policy=["public"]),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "train[95%]", "validation": "train[5%]", "test": "validation"}
        ),
        Rename(field="sentence1", to_field="text_a"),
        Rename(field="sentence2", to_field="text_b"),
        MapInstanceValues(
            mappers={"label": {"0": "entailment", "1": "not entailment"}}
        ),
        Set(fields={"classes": ["entailment", "not entailment"]}),
        Set(fields={"type_of_relation": "entailment"}),
        Set(fields={"text_a_type": "premise"}),
        Set(fields={"text_b_type": "hypothesis"}),
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
        "The Winograd Schema Challenge (Levesque et al., 2011) is a reading comprehension task in which a system must read a sentence with a pronoun and select the referent of that pronoun from a list of choices. The examples are manually constructed to foil simple statistical methods: Each one is contingent on contextual information provided by a single word or phrase in the sentence… See the full description on the dataset page: https://huggingface.co/datasets/nyu-mll/glue."
    ),
)

test_card(card)
add_to_catalog(card, "cards.wnli", overwrite=True)


card = TaskCard(
    loader=LoadHF(path="glue", name="wnli"),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "train[95%]", "validation": "train[5%]", "test": "validation"}
        ),
        Rename(field="sentence1", to_field="text_a"),
        Rename(field="sentence2", to_field="text_b"),
        MapInstanceValues(mappers={"label": {"0": "yes", "1": "no"}}),
        Set(fields={"classes": ["yes", "no"]}),
        Set(fields={"type_of_relation": "truthfulness"}),
        Set(fields={"text_a_type": "premise"}),
        Set(fields={"text_b_type": "hypothesis"}),
    ],
    task="tasks.classification.multi_class.relation",
    templates="templates.classification.multi_class.relation.truthfulness.all",
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
        "The Winograd Schema Challenge (Levesque et al., 2011) is a reading comprehension task in which a system must read a sentence with a pronoun and select the referent of that pronoun from a list of choices. The examples are manually constructed to foil simple statistical methods: Each one is contingent on contextual information provided by a single word or phrase in the sentence… See the full description on the dataset page: https://huggingface.co/datasets/nyu-mll/glue."
    ),
)

test_card(card)
add_to_catalog(card, "cards.wnli.truthfulness", overwrite=True)
