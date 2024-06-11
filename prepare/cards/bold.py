from unitxt import add_to_catalog
from unitxt.blocks import (
    InputOutputTemplate,
    LoadHF,
    Task,
    TaskCard,
    TemplatesList,
)
from unitxt.operators import (
    Copy,
    CopyFields,
    FilterByCondition,
    Set,
    Shuffle,
)
from unitxt.splitters import RenameSplits
from unitxt.struct_data_operators import DumpJson
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="AlexaAI/bold"),
    preprocess_steps=[
        RenameSplits(mapper={"train": "test"}),
        Set({"input_label": {}}),
        Copy(field="prompts/0", to_field="first_prompt"),
        Copy(field="wikipedia/0", to_field="first_wiki"),
        FilterByCondition(values={"domain": ["race", "gender"]}, condition="in"),
        Shuffle(page_size=10000),
        CopyFields(
            field_to_field={
                "first_prompt": "input_label/input",
                "category": "input_label/category",
                "first_wiki": "input_label/wiki",
            },
        ),
        DumpJson(field="input_label"),
    ],
    task=Task(
        inputs=["first_prompt"], outputs=["input_label"], metrics=["metrics.regard"]
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="{first_prompt}\n", output_format="{input_label}"
            ),
        ]
    ),
    __tags__={
        "arxiv": "2101.11718",
        "language": "en",
        "license": "cc-by-4.0",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "10K<n<100K",
        "source_datasets": "original",
        "task_categories": "text-generation",
    },
    __description__=(
        "Bias in Open-ended Language Generation Dataset (BOLD) is a dataset to evaluate fairness in open-ended language generation in English language. It consists of 23,679 different text generation prompts that allow fairness measurement across five domains: profession, gender, race, religious ideologies, and political ideologies… See the full description on the dataset page: https://huggingface.co/datasets/AlexaAI/bold."
    ),
)

test_card(card, demos_taken_from="test", strict=False)
add_to_catalog(card, "cards.bold", overwrite=True)
