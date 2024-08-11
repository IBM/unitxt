from unitxt import add_to_catalog
from unitxt.templates import InputOutputTemplateWithCustomTarget, TemplatesList


def add_rag_templates(templates_dict, task, out_field, reference_field="number_val"):
    rag_template_list = []
    for template_name, template_data in templates_dict.items():
        logprobs_suffixes = (
            [""] if "metric_template" in template_name else ["", "_logprobs"]
        )
        for logprobs_suffix in logprobs_suffixes:
            template_name_final = template_name + logprobs_suffix
            template = InputOutputTemplateWithCustomTarget(
                input_format=template_data["input"],
                output_format="{" + f"{out_field}" + "}",
                postprocessors=get_postprocessors_for_template(template_name_final),
                reference="{" + f"{reference_field}" + "}",
                target_prefix="Answer: ",
                instruction=template_data.get("instruction", ""),
            )
            full_template_name = f"templates.rag_eval.{task}.{template_name_final}"
            rag_template_list.append(full_template_name)
            add_to_catalog(template, full_template_name, overwrite=True)

    add_to_catalog(
        TemplatesList(rag_template_list),
        f"templates.rag_eval.{task}.all",
        overwrite=True,
    )


##################
# post processors
##################
judge_yes_no_postprocessors = [
    "processors.take_first_word",
    "processors.lower_case",
    "processors.yes_no_to_int",
    "processors.cast_to_float_return_zero_if_failed",
]

judge_yes_no_logprob_postprocessors = [
    "processors.load_json_from_predictions",
    "processors.bam_logprobs_to_yes_no_probs",
    "processors.cast_to_float_return_zero_if_failed",
]

judge_extract_pred_yes_no_postprocessors = [
    "processors.extract_from_double_brackets",
    "processors.lower_case",
    "processors.yes_no_to_int",
    "processors.cast_to_float_return_zero_if_failed",
]

judge_extract_pred_yes_no_logprob_postprocessors = [
    "processors.load_json_from_predictions",
    "processors.bam_last_token_logprobs_to_yes_no_probs",
    "processors.cast_to_float_return_zero_if_failed",
]

metric_postprocessors = [
    "processors.to_string_stripped",
    "processors.cast_to_float_return_zero_if_failed",
]


def get_postprocessors_for_template(template_name):
    if "metric_template" in template_name:
        return metric_postprocessors
    if "explain_first" in template_name:
        if "logprobs" in template_name:
            return judge_extract_pred_yes_no_logprob_postprocessors
        return judge_extract_pred_yes_no_postprocessors
    if "logprobs" in template_name:
        return judge_yes_no_logprob_postprocessors
    if "judge" in template_name:
        return judge_yes_no_postprocessors

    raise ValueError(f"Unsupported template name: {template_name}")


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

context_relevance_instructions = """You are given a question and the corresponding evidence. Please determine whether the evidence contain the answer to the question. Answer with only yes/no.\n
"""

context_relevance_instructions_ares = """Given the following question and document, you must analyze the provided document and determine whether it is sufficient for answering the question. In your evaluation, you should consider the content of the document and how it relates to the provided question. Answer with only yes/no.\n"""

correctness_instructions_instruct_qa_full = """System prompt: You are CompareGPT, a machine to verify the correctness of predictions. Answer with only yes/no.
You are given a question, the corresponding ground-truth answer and a prediction from a model. Compare the "Ground-truth answer" and the "Prediction" to determine whether the prediction correctly answers the question. All information in the ground-truth answer must be present in the prediction, including numbers and dates. You must answer "no" if there are any specific details in the ground-truth answer that are not mentioned in the prediction. There should be no contradicting statements in the prediction. The prediction may contain extra information. If the prediction states something as a possibility, treat it as a definitive answer.\n
"""

correctness_instructions_simplified = """You are given a question, the corresponding ground-truth answer and a prediction from a model. Compare the "Ground-truth answer" and the "Prediction" to determine whether the prediction correctly answers the question. All information in the ground-truth answer must be present in the prediction. Answer with only yes/no.\n"""


faithfilness_instructions_with_question_full = """System prompt: You are CompareGPT, a machine to verify the groundedness of predictions. Answer with only yes/no.
You are given a question, the corresponding evidence and a prediction from a model. Compare the "Prediction" and the "Evidence" to determine whether all the information of the prediction is present in the evidence or can be inferred from the evidence. You must answer "no" if there are any specific details in the prediction that are not mentioned in the evidence or cannot be inferred from the evidence.\n
"""

faithfilness_instructions_with_question_simplified = """You are given a question, the corresponding evidence and a prediction from a model. Compare the "Prediction" and the "Evidence" to determine whether all the information of the prediction is present in the evidence or can be inferred from the evidence. You must answer "no" if there are any specific details in the prediction that are not mentioned in the evidence or cannot be inferred from the evidence. Answer only "Yes" or "No".\n
"""

faithfulness_instructions_no_question_full = """System prompt: You are CompareGPT, a machine to verify the groundedness of predictions. Answer with only yes/no.
You are given a grounding evidence and a prediction from a model. Compare the "Prediction" and the "Evidence" to determine whether all the information of the prediction is present in the evidence or can be inferred from the evidence. You must answer "no" if there are any specific details in the prediction that are not mentioned in the evidence or cannot be inferred from the evidence.\n
"""

faithfulness_instructions_no_question_simplified = """You are given a grounding evidence and a prediction from a model. Compare the "Prediction" and the "Evidence" to determine whether all the information of the prediction is present in the evidence or can be inferred from the evidence. You must answer "no" if there are any specific details in the prediction that are not mentioned in the evidence or cannot be inferred from the evidence. Answer only "Yes" or "No".
"""
faithfulness_instructions_no_question_simplified_explained = """You are given an "Evidence" and a "Prediction" from a model. Compare the "Prediction" and the "Evidence" texts to determine whether all the information of the "Prediction" is present in the "Evidence" or can be inferred from the "Evidence". You must answer "No" if there are any specific details in the "Prediction" that are not mentioned in the "Evidence" or cannot be inferred from the "Evidence". Answer only "Yes" or "No". Then, provide an explanation to your answer.
"""

answer_relevance_instructions = """You are given a question and a prediction from a model. Please determine whether the prediction answers the question. Answer with only yes/no.\n
"""

correctness_referenceless_instructions_simple = """You are given a question, some corresponding evidence and a prediction from a model. Please determine whether the prediction is a correct and complete answer to the question given the provided evidence. Answer with only yes/no.\n"""

correctness_reference_based_with_context_instructions_simple = """You are given a question, some corresponding evidence, the ground truth answer and a prediction from a model. Please determine whether the prediction is a correct and complete answer to the question given the provided evidence and ground truth answer. Answer with only yes/no.\n"""

correctness_reference_based_no_context_instructions_simple = """You are given a question, the corresponding ground-truth answer and a prediction from a model. Compare the "Ground-truth answer" and the "Prediction" to determine whether the prediction is a correct and complete answer to the question. Answer with only yes/no.\n"""

correctness_referenceless_instructions_explain_first = (
    "You are given a question, some corresponding evidence and a prediction from a model. "
    "Please determine whether the prediction is a correct and complete answer to the question given the provided evidence. "
    "\nBegin your evaluation by generating your own answer to the question given the provided evidence. You must provide your answer before making the judgement."
    "\nWhen evaluating the model prediction, compare it with your answer. You must identify and correct any mistakes, partial information, or content that does not rely on the given evidence."
    "\nAfter providing your answer and explanation, you must output only one of the following choices as your final verdict with a label:"
    "\n\n1. The prediction is satisfactory and complete: [[Yes]]"
    "\n2. The prediction does not meet the requirements: [[No]]\n"
    '\n\nExample output: "Final verdict: [[Yes]]".\n'
)

####################
# Faithfulness dicts
####################
faithfulness_templates_dict = {
    "metric_template": {
        "input": "Question: {question}\nEvidence: {contexts}\n Prediction: {answer}",
    },
    "judge_with_question_full": {
        "input": faithfulness_with_question_input_format,
        "instruction": faithfilness_instructions_with_question_full,
    },
    "judge_with_question_simplified": {
        "input": faithfulness_with_question_input_format,
        "instruction": faithfilness_instructions_with_question_simplified,
    },
    "judge_no_question_full": {
        "input": faithfulness_no_question_input_format,
        "instruction": faithfulness_instructions_no_question_full,
    },
    "judge_no_question_simplified": {
        "input": faithfulness_no_question_input_format,
        "instruction": faithfulness_instructions_no_question_simplified,
    },
    "judge_no_question_simplified_explain": {
        "input": faithfulness_no_question_input_format,
        "instruction": faithfulness_instructions_no_question_simplified_explained,
    },
}
add_rag_templates(faithfulness_templates_dict, "faithfulness", "is_faithful")


####################
# Correctness dicts
####################
correctness_templates_dict = {
    "metric_template": {
        "input": "Question: {question}\nGround-truth answer: {ground_truths}\nPrediction: {answer}",
    },
    "judge_instruct_qa_format": {
        "input": correctness_input_format,
        "instruction": correctness_instructions_instruct_qa_full,
    },
    "judge_simplified_format": {
        "input": correctness_input_format,
        "instruction": correctness_instructions_simplified,
    },
    "judge_simplified_with_context": {
        "input": correctness_reference_based_with_context_input_format,
        "instruction": correctness_reference_based_with_context_instructions_simple,
    },
    "judge_simplified_no_context": {
        "input": correctness_input_format,
        "instruction": correctness_reference_based_no_context_instructions_simple,
    },
}
add_rag_templates(correctness_templates_dict, "correctness", "is_correct")

###################
# Context Relevance
###################
context_relevance_templates_dict = {
    "metric_template": {
        "input": "Question: {question}\nEvidence: {contexts}",
    },
    "judge_context_relevance": {
        "input": context_relevance_input_format,
        "instruction": context_relevance_instructions,
    },
    "judge_context_relevance_ares": {
        "input": context_relevance_input_format_ares,
        "instruction": context_relevance_instructions_ares,
    },
}
add_rag_templates(
    context_relevance_templates_dict, "context_relevance", "is_context_relevant"
)

###################
# Answer Relevance
###################
answer_relevance_templates_dict = {
    "metric_template": {
        "input": "Question: {question}\nPrediction: {answer}",
    },
    "judge_answer_relevance": {
        "input": answer_relevance_input_format,
        "instruction": answer_relevance_instructions,
    },
}
add_rag_templates(answer_relevance_templates_dict, "answer_relevance", "is_relevant")


###################
# Holistic Correctness (Reference-less)
###################
correctness_referenceless_templates_dict = {
    "metric_template": {
        "input": "Question: {question}\nEvidence: {contexts}\n Prediction: {answer}",
    },
    "judge_correctness_simple": {
        "input": correctness_referenceless_input_format,
        "instruction": correctness_referenceless_instructions_simple,
    },
    "judge_correctness_explain_first": {
        "input": correctness_referenceless_input_format,
        "instruction": correctness_referenceless_instructions_explain_first,
    },
}
add_rag_templates(
    correctness_referenceless_templates_dict, "correctness_holistic", "is_correct"
)
