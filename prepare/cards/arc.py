from datasets import load_dataset_builder

from prepare.cards.mmlu import MMLU_TEMPLATES, multiple_choice_preprocess
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
from src.unitxt.operators import RenameFields, JoinStr, TakeByField, ZipFieldValues, CopyFields, IndexOf
from unitxt.splitters import RenameSplits

# import huggingface_hub
# from huggingface_hub.hf_api import DatasetInfo as HFDatasetInfo, HfApi
# from huggingface_hub import DatasetFilter
# api = HfApi()
# analyzer = AnalyzerEngine()
# datasets = list(api.list_datasets(filter=DatasetFilter(dataset_name='cais/mmlu')))
# builder = load_dataset_builder(path='cais/mmlu')
from unitxt.templates import TemplatesDict

subtasks = ['ARC-Challenge', 'ARC-Easy']

for subtask in subtasks:
    # numbering=tuple(str(x) for x in range(200))
    numbering = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    expected_answer = 'number'  # 'number_and_answer' #'number'

    card = TaskCard(
        loader=LoadHF(path='ai2_arc', name=subtask),
        preprocess_steps=[
            AddFields({'topic': 'science'}),
            RenameFields({'answerKey': 'label', 'question': 'sentence1', 'choices': 'choices_struct'}),
            CopyFields(field_to_field={'choices_struct/text': 'choices', 'choices_struct/label': 'numbering'},
                       use_query=True),
            IndexOf(search_in='numbering', index_of='label', to_field='index'),

            *multiple_choice_preprocess(numbering='numbering', choices='choices', topic='topic', label_index='index')

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
