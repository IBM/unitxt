from datasets import load_dataset_builder

from prepare.cards.mmlu import multiple_choice_preprocess, MMLU_TEMPLATES
from src.unitxt.blocks import (
    LoadHF,
    SplitRandomMix,
    AddFields,
    TaskCard,
    NormalizeListFields,
    FormTask,
    TemplatesList,
    InputOutputTemplate,
    MapInstanceValues
)
from src.unitxt.test_utils.card import test_card

from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import RenameFields, JoinStr, TakeByField, ZipFieldValues, CopyFields, IndexOf, CastFields
from unitxt.splitters import RenameSplits

# import huggingface_hub
# from huggingface_hub.hf_api import DatasetInfo as HFDatasetInfo, HfApi
# from huggingface_hub import DatasetFilter
# api = HfApi()
# analyzer = AnalyzerEngine()
# datasets = list(api.list_datasets(filter=DatasetFilter(dataset_name='cais/mmlu')))
# builder = load_dataset_builder(path='cais/mmlu')
from unitxt.templates import TemplatesDict

subtasks = ['debiasex', 'l', 'm', 's', 'xl', 'xs']

for subtask in subtasks:
    # numbering=tuple(str(x) for x in range(200))
    numbering = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    expected_answer = 'number'  # 'number_and_answer' #'number'

    card = TaskCard(
        loader=LoadHF(path='winogrande', name=subtask),
        preprocess_steps=[
            AddFields({'topic': 'science', 'numbering':numbering}),
            ZipFieldValues(fields=['option1', 'option2'], to_field='choices'),
            CastFields(fields={"answer": "answer"}, cast_to="int"),
            *multiple_choice_preprocess(numbering='numbering', choices='choices', topic='topic', label_index='answer'),
        ],
        task=FormTask(
            inputs=['choices', 'sentence1', 'numbers', 'topic'],
            outputs=['label', ],
            metrics=['metrics.accuracy'],
        ),
        templates=MMLU_TEMPLATES
    )
    test_card(card)
    add_to_catalog(card, f'cards.ai2_arc.{subtask.replace("-", "_")}', overwrite=True)
