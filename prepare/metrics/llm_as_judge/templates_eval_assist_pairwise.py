from unitxt.templates import InputOutputTemplate

################################# Mixtral Templates #############################
assessment_prompt_mixtral = """<s> [INST]
You are provided a pair of responses (Response {option_a} and Response {option_b}) generated subject to a context. 
You will choose the better quality response subject to the quality criteria. 
This is the context:
{context_variables}
This is the quality criteria:
{criteria_name}
{criteria_description}
Response {option_a}:
{response_a}
Response {option_b}:
{response_b}
Keeping the quality criteria in mind, briefly assess which response is better.
Focus on the quality criteria during assessment, do not provide a general assessment.
[/INST]
Assessment:
"""

summarization_prompt_mixtral = """
<s> [INST]
Summarize the following assessment while maintaining the pertinent details.
Assessment: {assessment}
[/INST]
Assessment Summary:
"""

answer_prompt_mixtral = """
</s> 
[INST]
Now considering the quality criteria, which response is better quality?
Validate the answer against the assessment.
{options}
[/INST]
Answer: """

################################ LLAMA Templates ########################

assessment_prompt_llama = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are an fair and objective evaluator.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
You are provided a pair of responses (Response {option_a} and Response {option_b}) generated subject to a context. 
You will choose the better quality response subject to the quality criteria. 
This is the context:
{context_variables}
This is the quality criteria:
{criteria_name}
{criteria_description}
Response {option_a}:
{response_a}
Response {option_b}:
{response_b}
Keeping the quality criteria in mind, breifly assess which response is better.
Focus on the quality criteria during assessment, do not provide a general assessment.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Assessment: 
"""

summarization_prompt_llama = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Summarize the following assessment while maintaining the pertinent details.
Assessment: {assessment}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Assessment Summary:
"""

answer_prompt_llama = """<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Now considering the quality criteria, which response is better quality?
Validate the answer against the assessment.
{options}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Answer: """


################################### PROMETHEUS ###################################

assessment_prompt_prometheus = """###Task Description:
A context that includes information relevant to the nature or generation of the response, a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of two responses strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, choose a better response between Response {option_a} and Response {option_b}. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] ({option_a} or {option_b})"
4. Please do not generate any other opening, closing, and explanations.
###Context:
{context_variables}
###Response {option_a}:
{response_a}
###Response {option_b}:
{response_b}
###Score Rubric:
{rubric}
###Feedback: 
"""

pairwise_template_dict = {"mixtral_8_7b": {"assessment": InputOutputTemplate(input_format=assessment_prompt_mixtral), "summarization": InputOutputTemplate(input_format=summarization_prompt_mixtral), "answer" : InputOutputTemplate(input_format=answer_prompt_mixtral)},
                 "llama3_8b": {"assessment": InputOutputTemplate(input_format=assessment_prompt_llama), "summarization": InputOutputTemplate(input_format=summarization_prompt_llama), "answer" : InputOutputTemplate(input_format=answer_prompt_llama)},
                 "llama3_70b": {"assessment": InputOutputTemplate(input_format=assessment_prompt_llama), "summarization": InputOutputTemplate(input_format=summarization_prompt_llama), "answer" : InputOutputTemplate(input_format=answer_prompt_llama)},
                 "prometheus_8_7b": {"assessment": InputOutputTemplate(input_format=assessment_prompt_prometheus), "summarization": None, "answer" : None}
                }
