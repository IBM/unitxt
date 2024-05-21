from unitxt import add_to_catalog
from unitxt.blocks import (
    LoadHF,
    Task,
    TaskCard,
    TemplatesList,
)
from unitxt.operators import Shuffle
from unitxt.struct_data_operators import LoadJson
from unitxt.templates import MultiReferenceTemplate
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="akariasai/PopQA"),
    preprocess_steps=[
        Shuffle(page_size=14267),
        LoadJson(field="possible_answers"),
    ],
    task=Task(
        inputs=["question", "prop", "subj"],
        outputs=["possible_answers"],
        metrics=["metrics.accuracy"],
    ),
    templates=TemplatesList(
        [
            MultiReferenceTemplate(
                input_format="Answer to the following question. There is no need to explain the reasoning at all. "
                "Simply state just the answer in few words. No need for full answer. No need to repeat "
                "the question or words from the question. The answer text should be partial and contain "
                "only {prop}. Do not use full sentence. \nQuestion: {question}\nThe {prop} of {subj} is:",
                references_field="possible_answers",
                postprocessors=[
                    "processors.take_first_non_empty_line",
                    "processors.lower_case",
                ],
            ),
        ]
    ),
    __tags__={"croissant": True, "region": "us"},
    __description__=(
        "Dataset Card for PopQA\n"
        "Dataset Summary\n"
        "PopQA is a large-scale open-domain question answering (QA) dataset, consisting of 14k entity-centric QA pairs. Each question is created by converting a knowledge tuple retrieved from Wikidata using a template. Each question come with the original subject_entitiey, object_entityand relationship_type annotation, as well as Wikipedia monthly page views.\n"
        "Languages\n"
        "The dataset contains samples in English only.… See the full description on the dataset page: https://huggingface.co/datasets/akariasai/PopQA."
    ),
)

test_card(card, demos_taken_from="test", strict=False)
add_to_catalog(card, "cards.pop_qa", overwrite=True)
