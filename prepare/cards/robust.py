import json

from unitxt.blocks import Task, TaskCard, TemplatesList
from unitxt.loaders import LoadFromDictionary
from unitxt.operators import Apply, Copy, FeatureGroupedShuffle
from unitxt.templates import MultiReferenceTemplate
from unitxt.test_utils.card import test_card

loader = LoadFromDictionary(
    data={
        "test": [
            {
                "id": 1998386,
                "question": "Who is the producer of 'Five Children and It'?",
                "variant_id": 14,
                "variant_type": "paraphrase",
                "possible_answers": '["Samuel Hadida", "Lisa Henson", "Lisa Marie Henson"]',
                "int_docid": -9223194317102641503,
                "data_classification_policy": None,
            },
            {
                "id": 2053574,
                "question": "IN WHAT COUNTRY IS KANSAS?",
                "variant_id": 5,
                "variant_type": "simple",
                "possible_answers": '["United States of America", "the United States of America", "America", "U.S.A.", "US...erica", "U.S", "United States", "\'Murica"]',
                "int_docid": -9223013854673052067,
                "data_classification_policy": None,
            },
            {
                "id": 3157421,
                "question": "En Avant de Guingamp has what color?",
                "variant_id": 9,
                "variant_type": "paraphrase",
                "possible_answers": '["red", "red color"]',
                "int_docid": -9222912340912703729,
                "data_classification_policy": None,
            },
            {
                "id": 2933598,
                "question": "Who wrote The Ghost Road?",
                "variant_id": 8,
                "variant_type": "paraphrase",
                "possible_answers": '["Pat Barker"]',
                "int_docid": -9222792599468253924,
                "data_classification_policy": None,
            },
            {
                "id": 4904342,
                "question": "Who was the screenwriter for laaj?",
                "variant_id": 2,
                "variant_type": "simple",
                "possible_answers": '["Rauf Khalid", "Abdul Rauf Khalid"]',
                "int_docid": -9222584502299276841,
                "data_classification_policy": None,
            },
            {
                "id": 837389,
                "question": "What is the occupation of Herman A. Barnett?",
                "variant_id": 7,
                "variant_type": "paraphrase",
                "possible_answers": '["surgeon", "surgeons"]',
                "int_docid": -9222430282260774654,
                "data_classification_policy": None,
            },
            {
                "id": 288745,
                "question": "Who is the screenwriter of The Shout?",
                "variant_id": 9,
                "variant_type": "paraphrase",
                "possible_answers": '["Robert Graves", "Robert von Ranke-Graves", "Robert Von Ranke-Graves", "Robert Ranke...rt von Ranke Graves", "Jerzy Skolimowski"]',
                "int_docid": -9222420084410767468,
                "data_classification_policy": None,
            },
            {
                "id": 5035834,
                "question": "Luk√°≈° Bodeƒçek plays what sport?",
                "variant_id": 8,
                "variant_type": "paraphrase",
                "possible_answers": '["association football", "football", "soccer"]',
                "int_docid": -9222325131875051995,
                "data_classification_policy": None,
            },
            {
                "id": 4639545,
                "question": "Who was the producer of Jane",
                "variant_id": 4,
                "variant_type": "simple",
                "possible_answers": '["Oliver Morosco"]',
                "int_docid": -9222318534094783849,
                "data_classification_policy": None,
            },
            {
                "id": 1323260,
                "question": "WHO WAS THE PRODUCER OF THE SHINING?",
                "variant_id": 4,
                "variant_type": "simple",
                "possible_answers": '["Stephen King", "Stephen Edwin King", "Richard Bachman", "John Swithen"]',
                "int_docid": -9222228877568763454,
                "data_classification_policy": None,
            },
        ]
    }
)

instance_metric = "string_containment"
baseline_label = "original"
# the variants used as perturbations; this list can be extended if new ones are added
variant_label_values = ["simple", "paraphrase"]
# each label value on its own, vs original, and all variants together
variant_label_lists = [[variant] for variant in variant_label_values] + [
    variant_label_values
]
# the name to assign to the metric
# variant_names = variant_label_values + ["allvariants"]  Ruff
variant_names = [*variant_label_values, "allvariants"]

card = TaskCard(
    loader=loader,
    preprocess_steps=[
        FeatureGroupedShuffle(grouping_features=["id"], page_size=250000),
        Apply("possible_answers", function=json.loads, to_field="possible_answers"),
        # group_mean reductions expect grouping field called "group_id"
        Copy(field_to_field=[("id", "group_id")]),
    ],
    task=Task(
        inputs=["group_id", "id", "question", "variant_id", "variant_type"],
        outputs=["possible_answers"],
        metrics=["metrics.robustness.fixed_group_mean_string_containment"],
    ),
    templates=TemplatesList(
        [
            MultiReferenceTemplate(
                input_format="Question: {question}\nAnswer:",
                references_field="possible_answers",
                postprocessors=[
                    "processors.take_first_non_empty_line",
                    "processors.lower_case_till_punc",
                    "processors.to_string_stripped",
                ],
            ),
            MultiReferenceTemplate(
                input_format="Question: {question}\nI'm not certain, I think the answer is:",
                references_field="possible_answers",
                postprocessors=[
                    "processors.take_first_non_empty_line",
                    "processors.lower_case_till_punc",
                    "processors.to_string_stripped",
                ],
            ),
            MultiReferenceTemplate(
                input_format="Question: {question}\nI'm absolutely sure the answer is:",
                references_field="possible_answers",
                postprocessors=[
                    "processors.take_first_non_empty_line",
                    "processors.lower_case_till_punc",
                    "processors.to_string_stripped",
                ],
            ),
        ],
    ),
)

test_card(card, demos_taken_from="test", strict=True)
