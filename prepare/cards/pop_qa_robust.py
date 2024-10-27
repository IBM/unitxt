from unitxt import add_to_catalog
from unitxt.blocks import (
    LoadHF,
    Task,
    TaskCard,
)
from unitxt.operators import Apply, Copy, Rename
from unitxt.templates import MultiReferenceTemplate, TemplatesList
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="akariasai/PopQA"),
    preprocess_steps=[
        # FeatureGroupedShuffle(grouping_features=["id"], page_size=250000),
        # dafna: the above seems strange to me. I think "id" is unique per instance
        Apply("possible_answers", function="json.loads", to_field="possible_answers"),
        # group_mean reductions expect grouping field called "group_id"
        # dafna perhaps "prop_id" rather than the unique per instance "id"?
        Copy(field_to_field=[("prop_id", "group_id")]),
        # dafna: I arbitrary take fields to become the expected variant type and variant id
        Rename(field="obj", to_field="variant_id"),
        Rename(field="prop", to_field="variant_type"),
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
    __tags__={"region": "us"},
    __description__=(
        "PopQA is a large-scale open-domain question answering (QA) dataset, consisting of 14k entity-centric QA pairs. Each question is created by converting a knowledge tuple retrieved from Wikidata using a template. Each question come with the original subject_entitiey, object_entity, and relationship_type annotation, as well as Wikipedia monthly page views. Languages The dataset contains samples in English only.… See the full description on the dataset page: https://huggingface.co/datasets/akariasai/PopQA."
    ),
)

test_card(card, demos_taken_from="test", strict=True)
add_to_catalog(card, "cards.pop_qa_robust", overwrite=True)
