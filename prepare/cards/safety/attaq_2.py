from unitxt import add_to_catalog
from unitxt.blocks import (
    InputOutputTemplate,
    LoadHF,
    Task,
    TaskCard,
)
from unitxt.operators import Shuffle
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="ibm/AttaQ",
                  data_classification_policy=["public"]),
    preprocess_steps=[
        RenameSplits(mapper={"train": "test"}),
        Shuffle(page_size=2800),
    ],
    task=Task(
        input_fields={"input": str},
        reference_fields={"label": str},
        prediction_type=str,
        metrics=[
                 # "metrics.llm_as_judge.direct.watsonx.granite_guardian_3_0_2b"
                 "metrics.llm_as_judge.direct.watsonx.llama3_1_70b[context_fields=[input],criteria=metrics.llm_as_judge.direct.criterias.harmfulness,score_prefix=harmfulness_]",
                 "metrics.safety_metric[score_prefix=safety_]"],
    ),
    templates=[
        InputOutputTemplate(input_format="{input}\n", output_format=""),
        InputOutputTemplate(input_format="{input}", output_format=""),
    ],
)

test_card(card, strict=False, demos_taken_from="test", num_demos=0)
add_to_catalog(card, "cards.safety.attaq2", overwrite=True)
