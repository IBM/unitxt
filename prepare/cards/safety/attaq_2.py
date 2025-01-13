from unitxt import add_to_catalog
from unitxt.blocks import (
    InputOutputTemplate,
    LoadHF,
    Task,
    TaskCard,
)
from unitxt.operators import Copy, Set, Shuffle
from unitxt.splitters import RenameSplits
from unitxt.struct_data_operators import DumpJson
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="ibm/AttaQ",
                  data_classification_policy=["public"]),
    preprocess_steps=[
        RenameSplits(mapper={"train": "test"}),
        Shuffle(page_size=2800),
        Set({"input_label": {}}),
        Copy(
            field_to_field={"input": "input_label/input", "label": "input_label/label"}
        ),
        DumpJson(field="input_label"),
    ],
    task=Task(
        input_fields={"input": str},
        reference_fields={"input_label": str},
        prediction_type=str,
        metrics=["metrics.safety_metric", 
                 # "metrics.llm_as_judge.direct.watsonx.granite_guardian_3_0_2b"
                 "metrics.llm_as_judge.direct.watsonx.llama3_1_70b"
                 "[context_fields=[input,input_label],"
                 "criteria=metrics.llm_as_judge.direct.criterias.harmfulness]"],
        
    ),
    templates=[
        InputOutputTemplate(input_format="{input}\n", output_format="{input_label}"),
        InputOutputTemplate(input_format="{input}", output_format="{input_label}"),
    ],
    __description__="AttaQ",
    __tags__={
        "languages": ["english"],
        },
)

test_card(card, strict=False, demos_taken_from="test", num_demos=0)
add_to_catalog(card, "cards.safety.attaq2", overwrite=True)
