from typing import Union, List, Optional, Dict

from ..card import TaskCard
from ..instructions import InstructionsList, InstructionsDict
from ..loaders import Loader
from ..normalizers import NormalizeListFields
from ..operator import StreamingOperator
from ..operators import AddFields, MapInstanceValues, RenameFields
from ..task import FormTask
from ..templates import TemplatesDict, TemplatesList


def addClassificationChoices(label_name, label2string):
    return [MapInstanceValues(mappers={label_name: label2string}),
            AddFields(
                fields={
                    'choices': list(sorted(label2string.keys())),
                }
            ),
            NormalizeListFields(
                fields=['choices']
            ), ]


def create_2sentences_classification_card(loader: Loader, label_name: str, label2string: Dict, inputs: List[str],
                                          metrics: List[str] = tuple('accuracy'), task: FormTask = None,
                                          preprocess_steps: Optional[List[StreamingOperator]] = None,
                                          templates: Union[TemplatesList, TemplatesDict] = None,
                                          instructions: Union[InstructionsList, InstructionsDict] = None
                                          ) -> TaskCard:
    assert len(inputs) == 2, f"expected only 2 columns as input but received {inputs}"
    sentence1_col = "sentence1"
    sentence2_col = "sentence2"
    preprocess_steps += [
        *addClassificationChoices(label_name, label2string),
        RenameFields({inputs[0]: sentence1_col, inputs[1]: sentence2_col})
    ]
    if task is None:
        task = FormTask(
            inputs=['choices'] + [sentence1_col, sentence2_col],
            outputs=[label_name],
            metrics=metrics
        )
        return TaskCard(loader=loader, task=task, preprocess_steps=preprocess_steps, templates=templates,
                        instructions=instructions)


def create_sentence_classification_card(loader: Loader, label_name: str, label2string: Dict, inputs: List[str],
                                        metrics: List[str] = tuple('accuracy'), task: FormTask = None,
                                        preprocess_steps: Optional[List[StreamingOperator]] = None,
                                        templates: Union[TemplatesList, TemplatesDict] = None,
                                        instructions: Union[InstructionsList, InstructionsDict] = None
                                        ) -> TaskCard:
    """

    :param loader:
    :param label_name:
    :param label2string:
    :param inputs:
    :param metrics:
    :param task:
    :param preprocess_steps:
    :param templates:
    :param instructions:
    :return: TaskCard
    """
    # TODO labels should be deduced by default
    assert len(inputs) == 1, f"expected only 1 column as input but recieved {inputs}"
    sentence_col = "sentence1"
    preprocess_steps += [
        *addClassificationChoices(label_name, label2string),
        RenameFields({inputs[0]: sentence_col})
    ]
    if task is None:
        task = FormTask(
            inputs=['choices'] + [sentence_col],
            outputs=[label_name],
            metrics=metrics
        )
        return TaskCard(loader=loader, task=task, preprocess_steps=preprocess_steps, templates=templates,
                        instructions=instructions)
