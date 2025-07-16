from unitxt.blocks import (
    LoadHF,
    MapInstanceValues,
    Rename,
    Set,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.settings_utils import get_settings
from unitxt.struct_data_operators import ParseCSV
from unitxt.templates import InputOutputTemplate
from unitxt.test_utils.card import test_card

settings = get_settings()

with settings.context(allow_unverified_code=True):
    card = TaskCard(
        loader=LoadHF(
            path="wenhu/tab_fact",
            revision="refs/convert/parquet",
            data_dir="tab_fact",
            splits=["train", "test", "validation"],
            data_classification_policy=["public"],
        ),
        preprocess_steps=[
            ParseCSV(field="table_text", to_field="text_a", separator="#"),
            Rename(field="statement", to_field="text_b"),
            MapInstanceValues(mappers={"label": {"0": "refuted", "1": "entailed"}}),
            Set(
                fields={
                    "type_of_relation": "entailment",
                    "text_a_type": "Table",
                    "text_b_type": "Statement",
                    "classes": ["refuted", "entailed"],
                }
            ),
        ],
        task="tasks.classification.multi_class.relation",
        templates=[
            InputOutputTemplate(
                instruction="Given a {text_a_type} and {text_b_type} classify the {type_of_relation} of the {text_b_type} to one of {classes}."
                + "\nOutput only the final answer without any explanations, extra information, or introductory text."
                + "\nHere are some input-output examples. Read the examples carefully to figure out the mapping. The output of the last example is not given, and your job is to figure out what it is.",
                input_format="{text_a_type}: {text_a}\n{text_b_type}: {text_b} ",
                output_format="{label}",
                postprocessors=[
                    "processors.take_first_non_empty_line",
                    "processors.lower_case_till_punc",
                ],
            ),
        ],
        __tags__={
            "arxiv": "1909.02164",
            "license": "cc-by-4.0",
            "region": "us",
            "task_categories": "text-classification",
        },
        __description__=(
            "The problem of verifying whether a textual hypothesis holds the truth based on the given evidence, also known as fact verification, plays an important role in the study of natural language understanding and semantic representation. However, existing studies are restricted to dealing with unstructured textual evidence (e.g., sentences and passages, a pool of passages), while verification using structured forms of evidence, such as tables, graphs, and databases, remains unexplored. TABFACT is large scale dataset with 16k Wikipedia tables as evidence for 118k human annotated statements designed for fact verification with semi-structured evidence… See the full description on the dataset page: https://huggingface.co/datasets/ibm/tab_fact"
        ),
    )

    test_card(card, num_demos=2, demos_pool_size=20)
    add_to_catalog(card, "cards.tab_fact", overwrite=True)
