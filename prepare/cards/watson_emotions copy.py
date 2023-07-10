from src.unitxt.blocks import (
    LoadHF,
    SplitRandomMix,
    AddFields,
    TaskCard,
    NormalizeListFields,
    FormTask,
    TemplatesList,
    InputOutputTemplate,
)

from src.unitxt.catalog import add_to_catalog

task = FormTask(inputs=['text', 'textual_choices'], outputs=['labels'], metrics=['watson_single_label_multi_class'])

add_to_catalog(task, 'classification_task', overwrite=True)

card = TaskCard(
        loader=LoadHF(path='/dccstor/actr/unitxt-datasets/watson_emotion/'),
        preprocess_steps=[
            SplitRandomMix({'train': 'train[:95%]', 'validation': 'train[95%:]', 'test': 'test'}),
            AddFields(fields={'choices': ['fear', 'joy', 'anger', 'sadness', 'disgust', 'none']}),
            NormalizeListFields(fields=['choices'], key_prefix='textual_'),
        ],
        task='classification_task',
        templates='classification_templates'
    )

print(card.task)

add_to_catalog(card, 'watson_emotion_card', overwrite=True)


        