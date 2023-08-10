from datasets import load_dataset_builder
from prepare.cards.mmlu import (
    CONTEXT_MMLU_TEMPLATES,
    MMLU_TEMPLATES,
    multiple_choice_inputs_outputs,
    multiple_choice_preprocess,
)
from src.unitxt.blocks import (
    AddFields,
    FormTask,
    InputOutputTemplate,
    LoadHF,
    MapInstanceValues,
    NormalizeListFields,
    SplitRandomMix,
    TaskCard,
    TemplatesList,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import (
    CopyFields,
    IndexOf,
    JoinStr,
    ListFieldValues,
    RenameFields,
    ShuffleFieldValues,
    TakeByField,
    ZipFieldValues,
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
subsets = ["all", "high", "middle"]
for subset in subsets:
    card = TaskCard(
        loader=LoadHF(path="race", name=subset),
        preprocess_steps=[
            AddFields({"numbering": numbering, "topic": "reading comprehension"}),
            IndexOf(search_in="numbering", index_of="answer", to_field="index"),
            *multiple_choice_preprocess(
                context="article",
                question="question",
                numbering="numbering",
                choices="options",
                topic="topic",
                label_index="index",
            ),
        ],
        task=FormTask(
            **multiple_choice_inputs_outputs(context=True),
            metrics=["metrics.accuracy"],
        ),
        templates=CONTEXT_MMLU_TEMPLATES,
    )
    test_card(card)
    add_to_catalog(card, f"cards.race_{subset}", overwrite=True)
