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

from src.unitxt.catalog import add_to_catalog

card = TaskCard(
        loader=LoadHF(path='/dccstor/actr/unitxt-datasets/watson_emotion/'),
        preprocess_steps=[
            SplitRandomMix({'train': 'train[95%]', 'validation': 'train[5%]', 'test': 'test'}),
            AddFields(fields={'choices': ['fear', 'joy', 'anger', 'sadness', 'disgust', 'none']}),
            NormalizeListFields(fields=['choices'], key_prefix='textual_'),
            MapInstanceValues(mappers={'labels': {'[]': ['none']}}, strict=False),
        ],
        task=FormTask(inputs=['text', 'textual_choices'], outputs=['labels'], metrics=['accuracy']),
        templates=TemplatesList([
            InputOutputTemplate(input_format='{text}. Out of the following: {textual_choices}. The emotion expressed for the message is ', output_format='{labels}'),
            InputOutputTemplate(input_format='What is the emotion out of the following: {textual_choices}. In the text: {text}', output_format='{labels}')
        ])
    )

add_to_catalog(card, 'cards.watson_emotion_card', overwrite=True)


        