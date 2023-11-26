from typing import List, Union

from src.unitxt.blocks import AddFields, FormTask, LoadHF, TaskCard
from src.unitxt.catalog import add_to_catalog

# numbering=tuple(str(x) for x in range(200))
from src.unitxt.operator import StreamingOperator
from src.unitxt.operators import CastFields, JoinStr, RenameFields
from src.unitxt.templates import InputOutputTemplate, TemplatesDict
from src.unitxt.test_utils.card import test_card

answers = ["yes", "false"]
expected_answer = "number"  # 'number_and_answer' #'number'

templates = {
    "clean": """Question: {question}.\nAnswer:
                    """.strip(),
}

QA_TEMPLATES = TemplatesDict(
    {
        key: InputOutputTemplate(input_format=val, output_format="{label}")
        for key, val in templates.items()
    }
)

CONTEXT_QA_TEMPLATES = TemplatesDict(
    {
        key: InputOutputTemplate(
            input_format=val.replace(
                "Question:", "Context: {context}\nQuestion:"
            ).replace("{sentence1}", "{context}\n{sentence1}"),
            output_format="{label}",
        )
        for key, val in templates.items()
    }
)


def question_answering_outputs():
    return ["label"]


def question_answering_inputs_outputs(context=False, answers=False, topic=False):
    return {
        "inputs": question_answering_inputs(
            context=context, answers=answers, topic=topic
        ),
        "outputs": question_answering_outputs(),
    }


def question_answering_inputs(context=False, answers=False, topic=False):
    inputs = ["question", "label"]
    if context:
        inputs.append("context")
    if answers:
        inputs.append("answers")
    if topic:
        inputs.append("topic")
    return inputs


def question_answering_preprocess(
    question: str,
    answer: str,
    context: str = None,
    answers: str = None,
    topic: str = None,
) -> List[Union[StreamingOperator, str]]:
    """
    Processing to make a unified format of question answering.

    :param numbering: the field containing the numerals to use (e.g. ABCD [1,2,3,4])
    :param choices: the field with the choices (e.g. ['apple','bannana']
    :param topic: the field containing the topic of the question
    :param label_index: in what index is the right index (consider using IndexOf function if you have the answer instead)
    :param expected_answer: what format should the 'label' field be answer\number\number_and_answer
    :return:
    """
    renames = {
        "question": question,
        "label": answer,
        "context": context,
        "answers": answers,
        "topic": topic,
    }
    renames = {v: k for k, v in renames.items() if v}

    return [
        RenameFields(field_to_field=renames),
        JoinStr(separator=",", field=answers, to_field="answers"),
    ]


card = TaskCard(
    loader=LoadHF(path="boolq"),
    preprocess_steps=[
        "splitters.small_no_test",
        AddFields(
            {
                "topic": "boolean questions",
                "answers": answers,
            },
        ),
        CastFields(fields={"answer": "str"}),
        *question_answering_preprocess(
            context="passage",
            question="question",
            answers="answers",
            answer="answer",
            topic="topic",
        ),
    ],
    task=FormTask(
        **question_answering_inputs_outputs(topic=True, context=True),
        metrics=["metrics.accuracy"],
    ),
    templates=CONTEXT_QA_TEMPLATES,
)

test_card(card, demos_taken_from="test")
add_to_catalog(card, "cards.boolq", overwrite=True)
