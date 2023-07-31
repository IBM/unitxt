from datasets import load_dataset_builder

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
            AddFields({'topic': 'science'}),
            CastFields(field="answer", cast_to=int),
            RenameFields({'answerKey': 'label', 'sentence': 'sentence1', 'choices': 'choices_struct'}),
            CopyFields(field_to_field={'choices_struct/text': 'choices', 'choices_struct/label': 'numbering'},
                       use_query=True),
            ZipFieldValues(fields=['numbering', 'choices'], to_field='choices'),
            JoinStr(separator='. ', field='choices/*', to_field='choices_list', use_query=True,
                    process_every_value=True),
            IndexOf(search_in='numbering', index_of='label', to_field='index'),
            TakeByField(field='choices_list', index='index', to_field='number_and_answer'),
            TakeByField(field='numbering', index='index', to_field='number'),
            JoinStr(separator=',', field='choices/*/0', to_field='numbers', use_query=True),
            JoinStr(separator=' ', field='choices_list', to_field='choices'),  # field_to_field
            RenameFields({expected_answer: 'label'})
        ],
        task=FormTask(
            inputs=['choices', 'sentence1', 'numbers', 'topic'],
            outputs=['label', ],
            metrics=['metrics.accuracy'],
        ),
        templates=TemplatesDict({
            'original': InputOutputTemplate(
                input_format="""
                            The following are multiple choice questions (with answers) about {topic}.\n
                            {sentence1}.\nAnswers: {choices}.\nAnswer:
                    """.strip(),
                output_format='{label}',
            ),
            "helm": InputOutputTemplate(
                input_format="""
                            The following are multiple choice questions (with answers) about {topic}.\n\n
                            Question: {sentence1}.\nAnswers: {choices}.\nAnswer:
                    """.strip(),
                output_format='{label}',
            ),
            "lm_eval_harness": InputOutputTemplate(
                input_format="""
                            Question: {sentence1}.\nChoices:\n{choices}.\nAnswer:
                    """.strip(),
                output_format='{label}',
            ),
            "fm-eval": InputOutputTemplate(
                input_format="""
                            The following are multiple choice questions (with answers) about {topic}.\n\n
                            Question: {sentence1}.\nChoose from {numbers}\nAnswers: {choices}.\nAnswer:
                    """.strip(),
                output_format='{label}',
            ),
        })
    )
    test_card(card)
    add_to_catalog(card, f'cards.ai2_arc.{subtask.replace("-", "_")}', overwrite=True)
