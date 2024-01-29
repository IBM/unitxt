import json

from src.unitxt import add_to_catalog
from src.unitxt.blocks import (
    FormTask,
    LoadHF,
    TaskCard,
    TemplatesList,
)
from src.unitxt.operators import Apply, Shuffle
from src.unitxt.templates import MultiReferenceTemplate
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="akariasai/PopQA"),
    preprocess_steps=[
        Shuffle(page_size=14267),
        Apply("possible_answers", function=json.loads, to_field="possible_answers"),
    ],
    task=FormTask(
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
)

test_card(card, demos_taken_from="test", strict=False)
add_to_catalog(card, "cards.pop_qa", overwrite=True)
