import sys

from unitxt import add_to_catalog
from unitxt.card import TaskCard
from unitxt.collections_operators import DuplicateByList, Slice
from unitxt.loaders import LoadHF
from unitxt.operators import (
    Copy,
    IndexOf,
    ListFieldValues,
    MapInstanceValues,
    RenameFields,
    Set,
    Shuffle,
    ShuffleFieldValues,
)
from unitxt.string_operators import Join, Split
from unitxt.test_utils.card import test_card

gec_card = TaskCard(
    loader=LoadHF(
        path="grammarly/coedit",
        streaming=True,
        filtering_lambda="lambda x: x['task'] == 'gec'",
    ),
    preprocess_steps=[
        "splitters.small_no_test",
        Split(field="src", by=": "),
        Slice(field="src", start=1),
        Join(field="src", by=": "),
        RenameFields(field_to_field={"src": "original_text"}),
        ListFieldValues(fields=["tgt"], to_field="corrected_texts"),
    ],
    task="tasks.grammatical_error_correction",
    templates="templates.grammatical_error_correction.all",
    __tags__={
        "arxiv": "2305.09857",
        "language": "en",
        "license": "apache-2.0",
        "region": "us",
        "size_categories": "10K<n<100K",
        "task_categories": "text-generation",
    },
    __description__=(
        "This is the dataset that was used to train the CoEdIT text editing models. Full details of the dataset can be found in our paper… See the full description on the dataset page: https://huggingface.co/datasets/grammarly/coedit."
    ),
)

error_detection_metrics = [
    "metrics.accuracy",
    "metrics.f1_binary",
    "metrics.precision_binary",
    "metrics.recall_binary",
]

error_detection_card = TaskCard(
    loader=LoadHF(
        path="grammarly/coedit",
        streaming=True,
        filtering_lambda="lambda x: x['task'] == 'gec'",
    ),
    preprocess_steps=[
        "splitters.small_no_test",
        Split(field="src", by=": "),
        Slice(field="src", start=1),
        Join(field="src", by=": "),
        ListFieldValues(
            fields=["tgt", "src"],
            to_field="correct_and_incorrect",
        ),
        DuplicateByList(field="correct_and_incorrect", to_field="text"),
        IndexOf(index_of="text", search_in="correct_and_incorrect", to_field="label"),
        Set(
            fields={
                "class": "Grammatically incorrect",
            }
        ),
        Shuffle(page_size=sys.maxsize),
    ],
    task=f"tasks.classification.binary.zero_or_one[metrics=[{','.join(error_detection_metrics)}]]",
    templates="templates.grammatical_error_detection.all",
    __tags__={
        "arxiv": "2305.09857",
        "language": "en",
        "license": "apache-2.0",
        "region": "us",
        "size_categories": "10K<n<100K",
        "task_categories": "text-generation",
    },
    __description__=(
        "This is the dataset that was used to train the CoEdIT text editing models. Full details of the dataset can be found in our paper… See the full description on the dataset page: https://huggingface.co/datasets/grammarly/coedit."
    ),
)


test_card(gec_card, strict=False)
add_to_catalog(gec_card, "cards.coedit_gec", overwrite=True)

test_card(error_detection_card)
add_to_catalog(error_detection_card, "cards.coedit_error_detection", overwrite=True)


card = TaskCard(
    loader=LoadHF(path="grammarly/coedit", streaming=True),
    preprocess_steps=[
        Shuffle(page_size=sys.maxsize),
        "splitters.small_no_test",
        Split(field="src", by=": "),
        Slice(field="src", start=1),
        Copy(field="src/0", to_field="instruction"),
        Join(field="src", by=": "),
        ListFieldValues(
            fields=["tgt", "src"],
            to_field="choices",
        ),
        ShuffleFieldValues(field="choices"),
        Set(
            fields={
                "output_type": "sentence",
                "input_type": "sentence",
            }
        ),
        RenameFields(field="src", to_field="input"),
        IndexOf(search_in="choices", index_of="tgt", to_field="output_choice"),
    ],
    task="tasks.evaluation.preference",
    templates="templates.evaluation.preference.all",
    __tags__={
        "arxiv": "2305.09857",
        "language": "en",
        "license": "apache-2.0",
        "region": "us",
        "size_categories": "10K<n<100K",
        "task_categories": "text-generation",
    },
    __description__=(
        "This is the dataset that was used to train the CoEdIT text editing models. Full details of the dataset can be found in our paper… See the full description on the dataset page: https://huggingface.co/datasets/grammarly/coedit."
    ),
)

test_card(card)
add_to_catalog(card, "cards.coedit.preference", overwrite=True)

card = TaskCard(
    loader=LoadHF(
        path="grammarly/coedit",
        streaming=True,
        filtering_lambda="lambda x: x['task'] in ['gec', 'simplification', 'coherence', 'neutralize']",
    ),
    preprocess_steps=[
        Shuffle(page_size=sys.maxsize),
        "splitters.small_no_test",
        Split(field="src", by=": "),
        Slice(field="src", start=1),
        Join(field="src", by=": "),
        ListFieldValues(
            fields=["tgt", "src"],
            to_field="choices_texts",
        ),
        ShuffleFieldValues(field="choices_texts"),
        Copy(field="task", to_field="required_attribute"),
        MapInstanceValues(
            mappers={
                "required_attribute": {
                    "gec": "grammatically correct",
                    "simplification": "simple",
                    "coherence": "coherent",
                    "neutralize": "neutral",
                }
            }
        ),
        Copy(field="task", to_field="attribute_type"),
        MapInstanceValues(
            mappers={
                "attribute_type": {
                    "gec": "gramaticity",
                    "simplification": "simplicity",
                    "coherence": "coherence",
                    "neutralize": "neutrality",
                }
            }
        ),
        Set(
            fields={
                "choices_text_type": "sentences",
            }
        ),
        RenameFields(field_to_field={"tgt": "choice"}),
    ],
    task="tasks.selection.by_attribute",
    templates="templates.selection.by_attribute.all",
    __tags__={
        "arxiv": "2305.09857",
        "language": "en",
        "license": "apache-2.0",
        "region": "us",
        "size_categories": "10K<n<100K",
        "task_categories": "text-generation",
    },
    __description__=(
        "This is the dataset that was used to train the CoEdIT text editing models. Full details of the dataset can be found in our paper… See the full description on the dataset page: https://huggingface.co/datasets/grammarly/coedit."
    ),
)

test_card(card)
add_to_catalog(card, "cards.coedit.selection", overwrite=True)


card = TaskCard(
    loader=LoadHF(
        path="grammarly/coedit",
        streaming=True,
        filtering_lambda="lambda x: x['task'] in ['gec', 'simplification', 'coherence', 'neutralize']",
    ),
    preprocess_steps=[
        Shuffle(page_size=sys.maxsize),
        "splitters.small_no_test",
        Split(field="src", by=": "),
        Slice(field="src", start=1),
        Join(field="src", by=": "),
        Copy(field="task", to_field="required_attribute"),
        MapInstanceValues(
            mappers={
                "required_attribute": {
                    "gec": "grammatically correct",
                    "simplification": "simple",
                    "coherence": "coherent",
                    "neutralize": "neutral",
                }
            }
        ),
        Copy(field="task", to_field="attribute_type"),
        MapInstanceValues(
            mappers={
                "attribute_type": {
                    "gec": "gramaticity",
                    "simplification": "simplicity",
                    "coherence": "coherence",
                    "neutralize": "neutrality",
                }
            }
        ),
        Set(
            fields={
                "input_text_type": "sentence",
                "output_text_type": "sentence",
            }
        ),
        RenameFields(field_to_field={"tgt": "output_text", "src": "input_text"}),
    ],
    task="tasks.rewriting.by_attribute",
    templates="templates.rewriting.by_attribute.all",
    __tags__={
        "arxiv": "2305.09857",
        "language": "en",
        "license": "apache-2.0",
        "region": "us",
        "size_categories": "10K<n<100K",
        "task_categories": "text-generation",
    },
    __description__=(
        "This is the dataset that was used to train the CoEdIT text editing models. Full details of the dataset can be found in our paper… See the full description on the dataset page: https://huggingface.co/datasets/grammarly/coedit."
    ),
)

test_card(card)
add_to_catalog(card, "cards.coedit.rewriting", overwrite=True)


card = TaskCard(
    loader=LoadHF(
        path="grammarly/coedit",
        streaming=True,
        filtering_lambda="lambda x: x['task'] == 'paraphrase'",
    ),
    preprocess_steps=[
        "splitters.small_no_test",
        Split(field="src", by=": "),
        Slice(field="src", start=1),
        Join(field="src", by=": "),
        Set(
            fields={
                "text_type": "sentence",
            }
        ),
        RenameFields(field_to_field={"tgt": "output_text", "src": "input_text"}),
    ],
    task="tasks.rewriting.paraphrase",
    templates="templates.rewriting.paraphrase.all",
    __tags__={
        "arxiv": "2305.09857",
        "language": "en",
        "license": "apache-2.0",
        "region": "us",
        "size_categories": "10K<n<100K",
        "task_categories": "text-generation",
    },
    __description__=(
        "This is the dataset that was used to train the CoEdIT text editing models. Full details of the dataset can be found in our paper… See the full description on the dataset page: https://huggingface.co/datasets/grammarly/coedit."
    ),
)

test_card(card)
add_to_catalog(card, "cards.coedit.paraphrase", overwrite=True)
