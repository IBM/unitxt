from unitxt.eval_assist_constants import ModelFamilyEnum
from unitxt.templates import InputOutputTemplate


role_section = """
You are presented with a response generated subject to a context.
The context includes information relevant to the nature or generation of the response.
You will assess the quality of the response subject to an evaluation criteria.
"""

assessment_section = """
Briefly assess the quality of the response subject to the evaluation criteria.
Focus on the evaluation criteria during assessment, do not provide a general assessment.
Assessment: """

summary_section = """
Summarize the following assessment while maintaining the pertinent details.
"""

option_selection_section = """
Now consider the evaluation criteria and choose a final answer.
Validate the answer against the assessment.
"""

############################### Mixtral Templates #######################

assessment_prompt_mixtral = f"""<s> [INST]
{role_section}
###Context:
{{context_variables}}

###Response:
{{response}}

###Evaluation criteria:
{{criteria_description}}
{{display_options_instruction}}
{assessment_section}[/INST]
"""

summarization_prompt_mixtral = f"""<s> [INST]
{summary_section}
Assessment: {{assessment}}
Assessment Summary: [/INST]"""

answer_prompt_mixtral = f"""{{assessment_prompt}}{{assessment}}</s>
[INST]
{option_selection_section}
###Evaluation criteria:
{{criteria_description}}
{{score_option_instruction}}
Answer: [/INST]"""

############################################## LLAMA3 Templates ########################
assessment_prompt_llama =  f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are an fair and objective evaluator.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{role_section}
###Context:
{{context_variables}}
###Response:
{{response}}
###Evaluation criteria:
{{criteria_description}}
{{display_options_instruction}}
{assessment_section}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

summarization_prompt_llama = f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{summary_section}
Assessment: {{assessment}}
Assessment Summary: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

answer_prompt_llama =f"""{{assessment_prompt}}{{assessment}}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{option_selection_section}
###Evaluation criteria:
{{criteria_description}}
{{score_option_instruction}}
Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

################################ PROMETHEUS Templates ########################

assessment_prompt_prometheus = """###Task Description:
A context that includes information relevant to the nature or generation of the response, a response to evaluate, and a score rubric representing an evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, choose a score from the score rubric. Choose one of: {score_instructions}.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (Choose one of: {score_instructions})"
4. Please do not generate any other opening, closing, or explanations.
###Context:
{context_variables}
###Response to evaluate:
{response}
###Score Rubrics:
[{criteria_description}]
{score_option_instruction}
###Feedback:
"""

answer_prompt_prometheus = """{assessment_prompt}{assessment} [RESULT] """

############################################# GPT Templates #################################################

assessment_prompt_gpt = f"""{role_section}
###Context:
{{context_variables}}
###Response:
{{response}}
###Evaluation criteria:
{{criteria_description}}
{{display_options_instruction}}
{assessment_section}"""

summarization_prompt_gpt = f"""{summary_section}
Assessment: {{assessment}}
Assessment Summary:
"""

answer_prompt_gpt = f"""{{assessment_prompt}}{{assessment}}
{option_selection_section}
###Evaluation criteria:
{{criteria_description}}
{{score_option_instruction}}
Answer: """


################################### GRANITE Templates ###################################

assessment_prompt_granite = f"""<|system|>
You are Granite Chat, an AI language model developed by IBM. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.
<|user|>
{role_section}
Context:
{{context_variables}}

Response:
{{response}}

Evaluation criteria:
{{criteria_description}}
{{display_options_instruction}}
{assessment_section}<|assistant|>
"""

summarization_prompt_granite = f"""{summary_section}
Assessment: {{assessment}}

Assessment Summary: 
"""

answer_prompt_granite = f"""{{assessment_prompt}}{{assessment}}<|user|>
{option_selection_section}
Evaluation criteria:
{{criteria_description}}
{{score_option_instruction}}

Answer: <|assistant|>"""


direct_assessment_template_dict = {
    ModelFamilyEnum.MIXTRAL: {
        "assessment": InputOutputTemplate(input_format=assessment_prompt_mixtral), 
        "summarization": InputOutputTemplate(input_format=summarization_prompt_mixtral),
        "answer" : InputOutputTemplate(input_format=answer_prompt_mixtral)
    },
    ModelFamilyEnum.GRANITE: {
        "assessment": InputOutputTemplate(input_format=assessment_prompt_granite), 
        "summarization": InputOutputTemplate(input_format=summarization_prompt_granite),
        "answer" : InputOutputTemplate(input_format=answer_prompt_granite)
    },
    ModelFamilyEnum.LLAMA3: {
        "assessment": InputOutputTemplate(input_format=assessment_prompt_llama), 
        "summarization": InputOutputTemplate(input_format=summarization_prompt_llama),
        "answer" : InputOutputTemplate(input_format=answer_prompt_llama)
    },
    ModelFamilyEnum.GPT: {
        "assessment": InputOutputTemplate(input_format=assessment_prompt_gpt), 
        "summarization": InputOutputTemplate(input_format=summarization_prompt_gpt),
        "answer" : InputOutputTemplate(input_format=answer_prompt_gpt)
    },
    ModelFamilyEnum.PROMETHEUS: {
        "assessment": InputOutputTemplate(input_format=assessment_prompt_prometheus), 
        "summarization": None, 
        "answer": InputOutputTemplate(input_format=answer_prompt_prometheus)
    }
}
