from unitxt import add_to_catalog
from unitxt.templates import InputOutputTemplateWithCustomTarget

##################
# post processors
##################
judge_post_processors_dict = {
    "numeric": [
        "processors.take_first_word",
        "processors.lower_case",
        "processors.cast_to_float_return_zero_if_failed",
        "processors.scale_0_10_to_0_1",
    ],
    "verbal": [
        "processors.take_first_word",
        "processors.lower_case",
        "processors.extract_verbal_judgement",
    ],
    "verbal_good_bad": [
        "processors.extract_from_double_brackets",
        "processors.extract_verbal_judgement_bad_good",
        "processors.cast_to_float_return_zero_if_failed",
    ],
}


def add_rag_templates(
    templates_dict,
    task,
    out_field="textual_label",
    reference_field="number_val",
):
    rag_template_list = []
    for template_name, template_data in templates_dict.items():
        template_type = template_data.get("template_type", "numeric")
        post_processors = judge_post_processors_dict[template_type]
        template = InputOutputTemplateWithCustomTarget(
            input_format=template_data["input"],
            output_format="{" + f"{out_field}" + "}",
            postprocessors=post_processors,
            reference="{" + f"{reference_field}" + "}",
            target_prefix="Answer: ",
            instruction=template_data.get("instruction", ""),
        )
        full_template_name = f"templates.rag_eval.{task}.{template_name}"
        rag_template_list.append(full_template_name)
        add_to_catalog(template, full_template_name, overwrite=True)


########################################
# input formats (prediction always last)
########################################

correctness_input_format = """Question: {question}\n
Ground-truth answer: {ground_truths}\n
Prediction: {answer}
"""

correctness_reference_based_with_context_input_format = """Question: {question}\n
Evidence: {contexts}\n
Ground Truth: {ground_truths}
Prediction: {answer}
"""


faithfulness_with_question_input_format = """Question: {question}\n
Evidence: {contexts}\n
Prediction: {answer}
"""

faithfulness_no_question_input_format = """Evidence: {contexts}\n
Prediction: {answer}
"""

context_relevance_input_format = """Question: {question}\n
Evidence: {contexts}
"""

context_relevance_input_format_ares = """Question: {question}\n
Document: {contexts}
"""

answer_relevance_input_format = """Question: {question}\n
Prediction: {answer}
"""

correctness_referenceless_input_format = """Question: {question}\n
Evidence: {contexts}\n
Prediction: {answer}
"""


###############
# Instructions
###############


context_relevance_instructions_ares = """Given the following question and document, you must analyze the provided document and determine to what extent it is sufficient for answering the question. In your evaluation, you should consider the content of the document and how it relates to the provided question.
On a scale of 0 to 10, to what extent is the document sufficient for answering the question? Reply with your rating score without any preceding explanation.\n"""

context_relevance_instructions_ares_verbal_good_bad = """Given the following question and document, you must analyze the provided document and determine to what extent it is sufficient for answering the question. In your evaluation, you should consider the content of the document and how it relates to the provided question.
Provide a rating from one of the following choices: 'Very Bad', 'Bad', 'Mediocre', 'Good', 'Very Good'. Reply using the format of [[rating]], for example: '[[Mediocre]]'.
"""

context_relevance_instructions_ares_verbal = """Given the following question and document, you must analyze the provided document and determine to what extent it is sufficient for answering the question. In your evaluation, you should consider the content of the document and how it relates to the provided question.
Reply with one of the 4 options, without any further explanations:
"Completely Relevant" - if the question is completely answerable from the document.
"Mostly Relevant" - if the document contains most of the information required to answer the question but not enough to fully address it.
"Somewhat Relevant" - If the document is related to the question but not sufficient to answer it.
"Not Relevant" - If the document is irrelevant to the question.
"""

faithfilness_instructions_with_question_simplified = """You are given a question, the corresponding evidence and a prediction from a model. Compare the "Prediction" and the "Evidence" to determine to what extent the prediction is grounded in the evidence.
To be grounded in the evidence, all the information of the prediction must either be present in the evidence or deducible from the evidence.\n
The question is only given for context, and is irrelevant for determining the groundedness of the prediction.
On a scale of 0 to 10, to what extent is the prediction grounded in the evidence? Reply with your rating score without any preceding explanation.\n
"""


faithfulness_instructions_no_question_simplified = """You are given a grounding evidence and a prediction from a model. Compare the "Prediction" and the "Evidence" to determine to what extent the prediction is grounded in the evidence.
To be grounded in the evidence, all the information of the prediction must either be present in the evidence or deducible from the evidence.\n
On a scale of 0 to 10, to what extent is the prediction grounded in the evidence? Reply with your rating score without any preceding explanation.\n
"""

faithfilness_instructions_with_question_simplified_category = """You are given a question, the corresponding evidence and a prediction from a model. Compare the "Prediction" and the "Evidence" to determine to what extent the prediction is grounded in the evidence.
To be grounded in the evidence, all the information of the prediction must either be present in the evidence or deducible from the evidence.\n
Base your answer only on the information in the evidence. If the prediction is correct but not present in the evidence - it is not grounded.
The question is only given for context, and is irrelevant for determining the groundedness of the prediction.
Reply with one of the 4 options, without any further explanations:
"Completely Grounded" - if the prediction is grounded in the evidence.
"Mostly grounded" - if the vast majority of the information in the prediction is grounded in the evidence, but there is a small or negligible part of the prediction which is not present in the evidence.
"Somewhat grounded" - If some of the information in the prediction is grounded in the evidence.
"Not grounded" - If most or all of the information in the prediction is not grounded in the evidence
"""

faithfulness_instructions_no_question_simplified_category = """You are given a grounding evidence and a prediction from a model. Compare the "Prediction" and the "Evidence" to determine to what extent the prediction is grounded in the evidence.
To be grounded in the evidence, all the information of the prediction must either be present in the evidence or deducible from the evidence.\n
Base your answer only on the information in the evidence. If the prediction is correct but not present in the evidence than it is not grounded.
Reply with one of the 4 options, without any further explanations:
"Completely Grounded" - if the prediction is grounded in the evidence.
"Mostly grounded" - if the vast majority of the information in the prediction is grounded in the evidence, but there is a small or negligible part of the prediction which is not present in the evidence.
"Somewhat grounded" - If some of the information in the prediction is grounded in the evidence.
"Not grounded" - If most or all of the information in the prediction is not grounded in the evidence
"""

faithfilness_instructions_with_question_simplified_verbal_good_bad = """You are given a question, the corresponding evidence and a prediction from a model.
Compare the "Prediction" and the "Evidence" to determine whether or not the prediction is grounded in the evidence.
A good prediction is a prediction which is grounded in the evidence, i.e. all the information of the prediction must either be present in the evidence or deducible from the evidence.\n
Base your answer only on the information in the evidence. If the prediction is correct but not present in the evidence - it is not grounded.
The question is only given for context, and is irrelevant for determining the groundedness of the prediction.
Provide a rating from one of the following choices: 'Very Bad', 'Bad', 'Mediocre', 'Good', 'Very Good'. Reply using the format of [[rating]], for example: '[[Mediocre]]'.
"""

faithfulness_instructions_no_question_simplified_verbal_good_bad = """You are given a grounding evidence and a prediction from a model.
Compare the "Prediction" and the "Evidence" to determine whether or not the prediction is grounded in the evidence.
A good prediction is a prediction which is grounded in the evidence, i.e. all the information of the prediction must either be present in the evidence or deducible from the evidence.\n
Base your answer only on the information in the evidence. If the prediction is correct but not present in the evidence - it is not grounded.
Provide a rating from one of the following choices: 'Very Bad', 'Bad', 'Mediocre', 'Good', 'Very Good'. Reply using the format of [[rating]], for example: '[[Mediocre]]'.
"""

answer_relevance_instructions = """You are given a question and a prediction from a model. Please determine whether or not the prediction answers the question.
Provide a rating from one of the following choices: 'Very Bad', 'Bad', 'Mediocre', 'Good', 'Very Good'. Reply using the format of [[rating]], for example: '[[Mediocre]]'.
"""

answer_relevance_instructions_numeric = """You are given a question and a prediction from a model. Please determine whether or not the prediction answers the question.
The prediction answers the question if it provides a relevant, appropriate and and complete answer to the question.
Ignore the correctness of the prediction in your judgement.
On a scale of 0 to 10, to what extent does the prediction answer the question? Reply with your rating score without any preceding explanation.\n"""

answer_relevance_instructions_verbal_good_bad = """You are given a question and a prediction from a model. Please determine to what extent, on a scale of 0 to 10, the prediction answers the question.
Reply with your rating score without any preceding explanation.\n"""

correctness_referenceless_instructions_simple = """You are given a question, some corresponding evidence and a prediction from a model. Please determine whether the prediction is a correct and complete answer to the question given the provided evidence.\n
On a scale of 0 to 10, to what extent does the prediction answer the question? Reply with your rating score without any preceding explanation.\n"""

correctness_reference_based_no_context_instructions_simple = """You are given a question, the corresponding ground-truth answer and a prediction from a model. Compare the "Ground-truth answer" and the "Prediction" to determine whether the prediction is a correct and complete answer to the question.On a scale of 0 to 10, to what extent does the prediction answer the question? Reply with your rating score without any preceding explanation.\n"""

correctness_reference_based_no_context_instructions_loose = """You are given a question, the corresponding ground-truth answer and a prediction from a model. Compare the "Ground-truth answer" and the "Prediction" to determine whether the prediction correctly answers the question.
There should be no contradicting statements in the prediction. The prediction may contain extra information. If the prediction states something as a possibility, treat it as a definitive answer.
The prediction must contain all the important information presented in the ground truths, but doesn't have to fully match it.
On a scale of 0 to 10, to what extent does the prediction answer the question? Reply with your rating score without any preceding explanation.\n"""

correctness_reference_based_no_context_instructions_loose_verbal_good_bad = """You are given a question, the corresponding ground-truth answer and a prediction from a model. Compare the "Ground-truth answer" and the "Prediction" to determine whether the prediction correctly answers the question.
There should be no contradicting statements in the prediction. The prediction may contain extra information. If the prediction states something as a possibility, treat it as a definitive answer.
The prediction must contain all the important information presented in the ground truths, but doesn't have to fully match it.
Provide a rating from one of the following choices: 'Very Bad', 'Bad', 'Mediocre', 'Good', 'Very Good'. Reply using the format of [[rating]], for example: '[[Mediocre]]'.
"""

correctness_reference_based_no_context_instructions_loose_verbal = """You are given a question, the corresponding ground-truth answer and a prediction from a model. Compare the "Ground-truth answer" and the "Prediction" to determine whether the prediction correctly answers the question.
There should be no contradicting statements in the prediction. The prediction may contain extra information. If the prediction states something as a possibility, treat it as a definitive answer.
The prediction must contain all the important information presented in the ground truths, but doesn't have to fully match it.
Reply with one of the 4 options, without any further explanations:
"Completely Correct" - if the prediction provides the correct answer.
"Mostly Correct" - if the prediction provides the correct answer with respect to the ground truths, but is missing or adding some non-crucial information.
"Somewhat Correct" - If the answer is partially correct with respect to the ground truths, but misses on important information that is necessary to answer the question or adding subtantial information that is not present in the ground truth.
"Not Correct" - If the prediction is completely wrong, contradictory or irrelevant to the ground truth.
"""

####################
# Correctness dicts
####################
correctness_templates_dict = {
    "judge_loose_match_no_context_numeric": {
        "input": correctness_input_format,
        "instruction": correctness_reference_based_no_context_instructions_loose,
        "template_type": "numeric",
    },
    "judge_loose_match_no_context_verbal_good_bad": {
        "input": correctness_input_format,
        "instruction": correctness_reference_based_no_context_instructions_loose_verbal_good_bad,
        "template_type": "verbal_good_bad",
    },
    "judge_loose_match_no_context_verbal": {
        "input": correctness_input_format,
        "instruction": correctness_reference_based_no_context_instructions_loose_verbal,
        "template_type": "verbal",
    },
}
add_rag_templates(correctness_templates_dict, "answer_correctness", "is_correct")

###################
# Context Relevance
###################
context_relevance_templates_dict = {
    "judge_context_relevance_ares_numeric": {
        "input": context_relevance_input_format_ares,
        "instruction": context_relevance_instructions_ares,
        "template_type": "numeric",
    },
    "judge_context_relevance_ares_verbal_good_bad": {
        "input": context_relevance_input_format_ares,
        "instruction": context_relevance_instructions_ares_verbal_good_bad,
        "template_type": "verbal_good_bad",
    },
    "judge_context_relevance_ares_verbal": {
        "input": context_relevance_input_format_ares,
        "instruction": context_relevance_instructions_ares_verbal,
        "template_type": "verbal",
    },
}
add_rag_templates(
    context_relevance_templates_dict, "context_relevance", "is_context_relevant"
)

###################
# Answer Relevance
###################
answer_relevance_templates_dict = {
    "judge_answer_relevance_numeric": {
        "input": answer_relevance_input_format,
        "instruction": answer_relevance_instructions_numeric,
        "template_type": "numeric",
    },
    "judge_answer_relevance_verbal_good_bad": {
        "input": answer_relevance_input_format,
        "instruction": answer_relevance_instructions_verbal_good_bad,
        "template_type": "verbal_good_bad",
    },
}
add_rag_templates(answer_relevance_templates_dict, "answer_relevance", "is_relevant")


###################
# Holistic Correctness (Reference-less)
###################
correctness_referenceless_templates_dict = {
    "judge_correctness_simple_numeric": {
        "input": correctness_referenceless_input_format,
        "instruction": correctness_referenceless_instructions_simple,
        "template_type": "numeric",
    },
}

add_rag_templates(
    correctness_referenceless_templates_dict, "correctness_holistic", "is_correct"
)

faithfulness_templates_dict = {
    "judge_with_question_simplified_numeric": {
        "input": faithfulness_with_question_input_format,
        "instruction": faithfilness_instructions_with_question_simplified,
        "template_type": "numeric",
    },
    "judge_no_question_simplified_numeric": {
        "input": faithfulness_no_question_input_format,
        "instruction": faithfulness_instructions_no_question_simplified,
        "template_type": "numeric",
    },
    "judge_with_question_simplified_verbal": {
        "input": faithfulness_with_question_input_format,
        "instruction": faithfilness_instructions_with_question_simplified_category,
        "template_type": "verbal",
    },
    "judge_no_question_simplified_verbal": {
        "input": faithfulness_no_question_input_format,
        "instruction": faithfulness_instructions_no_question_simplified_category,
        "template_type": "verbal",
    },
    "judge_with_question_simplified_verbal_good_bad": {
        "input": faithfulness_with_question_input_format,
        "instruction": faithfilness_instructions_with_question_simplified_verbal_good_bad,
        "template_type": "verbal_good_bad",
    },
    "judge_no_question_simplified_verbal_good_bad": {
        "input": faithfulness_no_question_input_format,
        "instruction": faithfulness_instructions_no_question_simplified_verbal_good_bad,
        "template_type": "verbal_good_bad",
    },
}
add_rag_templates(faithfulness_templates_dict, "faithfulness", "is_faithful")
