from prepare.cards.mmlu_old import multiple_choice_preprocess
from src.unitxt.blocks import (
    AddFields,
    FormTask,
    LoadHF,
    TaskCard,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import (
    ListFieldValues,
)
from src.unitxt.test_utils.card import test_card

# import huggingface_hub
# from huggingface_hub.hf_api import DatasetInfo as HFDatasetInfo, HfApi
# from huggingface_hub import DatasetFilter
# api = HfApi()
# analyzer = AnalyzerEngine()
# datasets = list(api.list_datasets(filter=DatasetFilter(dataset_name='cais/mmlu')))
# builder = load_dataset_builder(path='cais/mmlu')

# numbering=tuple(str(x) for x in range(200))
numbering = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
expected_answer = "number"  # 'number_and_answer' #'number'

card = TaskCard(
    loader=LoadHF(path="piqa"),
    preprocess_steps=[
        AddFields({"numbering": numbering, "topic": "physical commonsense"}),
        ListFieldValues(fields=["sol1", "sol2"], to_field="choices"),
        # ZipFieldValues(fields=["sol1", "sol2"], to_field="choices"),
        *multiple_choice_preprocess(
            question="goal",
            numbering="numbering",
            choices="choices",
            topic="topic",
            label_index="label",
        ),
    ],
    task=FormTask(
        inputs=["choices", "sentence1", "numbers", "topic"],
        outputs=[
            "label",
        ],
        metrics=["metrics.accuracy"],
    ),
    templates="templates.qa.multiple_choice.original.all",
)
test_card(card)
add_to_catalog(card, "cards.piqa", overwrite=True)
