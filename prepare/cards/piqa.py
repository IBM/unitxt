from datasets import load_dataset_builder
from prepare.cards.mmlu import MMLU_TEMPLATES, multiple_choice_preprocess
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
    RenameFields,
    TakeByField,
    ZipFieldValues,
)
from src.unitxt.test_utils.card import test_card
from unitxt.splitters import RenameSplits

# import huggingface_hub
# from huggingface_hub.hf_api import DatasetInfo as HFDatasetInfo, HfApi
# from huggingface_hub import DatasetFilter
# api = HfApi()
# analyzer = AnalyzerEngine()
# datasets = list(api.list_datasets(filter=DatasetFilter(dataset_name='cais/mmlu')))
# builder = load_dataset_builder(path='cais/mmlu')
from unitxt.templates import TemplatesDict

subtasks = ["ARC-Challenge", "ARC-Easy"]

for subtask in subtasks:
    # numbering=tuple(str(x) for x in range(200))
    numbering = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    expected_answer = "number"  # 'number_and_answer' #'number'

    card = TaskCard(
        loader=LoadHF(path="piqa", name=subtask),
        preprocess_steps=[
            AddFields({"numbering": numbering, "topic": "physical commonsense"}),
            ZipFieldValues(fields=["sol1", "sol2"], to_field="choices"),
            RenameFields({"goal": "sentence1"}),
            *multiple_choice_preprocess(numbering="numbering", choices="choices", topic="topic", label_index="index"),
            # ZipFieldValues(fields=['numbering', 'choices'], to_field='choices'),
            #
            # JoinStr(separator='. ', field='choices/*', to_field='choices_list', use_query=True,
            #         process_every_value=True),
            # IndexOf(search_in='numbering', index_of='label', to_field='index'),
            # TakeByField(field='choices_list', index='index', to_field='number_and_answer'),
            # TakeByField(field='numbering', index='index', to_field='number'),
            # JoinStr(separator=',', field='choices/*/0', to_field='numbers', use_query=True),
            # JoinStr(separator=' ', field='choices_list', to_field='choices'),  # field_to_field
            RenameFields({expected_answer: "label"}),
        ],
        task=FormTask(
            inputs=["choices", "sentence1", "numbers", "topic"],
            outputs=[
                "label",
            ],
            metrics=["metrics.accuracy"],
        ),
        templates=MMLU_TEMPLATES,
    )
    test_card(card)
    add_to_catalog(card, f'cards.ai2_arc.{subtask.replace("-", "_")}', overwrite=True)
