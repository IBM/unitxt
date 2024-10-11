from unitxt.templates import InputOutputTemplate

############################### Mixtral Templates #######################

assessment_prompt_mixtral = """<s> [INST]
You are presented with a response generated subject to a context.
The context includes information relevant to the nature or generation of the response.
You will assess the quality of the response subject to an evaluation criteria.

###Context:
{context_variables}

###Response:
{response}

###Evaluation criteria:
{criteria}
{options}

Briefly assess the quality of the response subject to the evaluation criteria.
Focus on the evaluation criteria during assessment, do not provide a general assessment.
[/INST]
Assessment:
"""

summarization_prompt_mixtral = """<s> [INST]
Summarize the following assessment while maintaining the pertinent details.
Assessment: {assessment}
[/INST]
Assessment Summary:
"""

answer_prompt_mixtral = """</s>
[INST]
Now consider the evaluation criteria and choose a final answer.
Validate the answer against the assessment.
###Evaluation criteria:
{{criteria}}
{{options}}
[/INST]
Answer:"""

############################################Granite Templates########################

assessment_prompt_granite = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are an fair and objective evaluator.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
You are presented with a response generated subject to a context. 
The context includes information relevant to the nature or generation of the response.
You will assess the quality of the response subject to an evaluation criteria.
###Context:
{context_variables}
###Response:
{response}
###Evaluation criteria:
{criteria}
{options}
Briefly assess the quality of the response subject to the evaluation criteria.
Focus on the evaluation criteria during assessment, do not provide a general assessment.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Assessment: 
"""

summarization_prompt_granite = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Summarize the following assessment while maintaining the pertinent details.
Assessment: {assessment}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Assessment Summary:
"""

answer_prompt_granite = """<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Now consider the evaluation criteria and choose a final answer.
Validate the answer against the assessment.
###Evaluation criteria:
{criteria}
{options}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Answer: """

##############################################LLAMA3 Templates########################
assessment_prompt_llama =  """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are an fair and objective evaluator.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
You are presented with a response generated subject to a context. 
The context includes information relevant to the nature or generation of the response.
You will assess the quality of the response subject to an evaluation criteria.
###Context:
{context_variables}
###Response:
{response}
###Evaluation criteria:
{criteria}
{options}
Briefly assess the quality of the response subject to the evaluation criteria.
Focus on the evaluation criteria during assessment, do not provide a general assessment.<|eot_id|>
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
Now consider the evaluation criteria and choose a final answer.
Validate the answer against the assessment.
###Evaluation criteria:
{criteria}
{options}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Answer: """

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
[{criteria}]
{score_rubric}
###Feedback:
"""

############################################# GPT #################################################

assessment_prompt_gpt = """You are presented with a response generated to satisfy an instruction. 
You will assess the quality of the response subject to an evaluation criteria.
###Context:
{context_variables}
###Response:
{response}
###Evaluation criteria:
{criteria}
{options}
Briefly assess the quality of the response subject to the evaluation criteria.
Focus on the evaluation criteria during assessment, do not provide a general assessment.
Assessment: 
"""

summarization_prompt_gpt = """Summarize the following assessment into an single easy to understand statement.
Assessment: {assessment}
Assessment Summary:
""" 

answer_prompt_gpt = """Now consider the evaluation criteria and choose a final answer. 
Validate the answer against the assessment.
###Evaluation criteria:
{criteria}
{options}
Answer: """

template_dict = {"mixtral": {"assessment": InputOutputTemplate(input_format=assessment_prompt_mixtral), "summarization": InputOutputTemplate(input_format=summarization_prompt_mixtral), "answer" : InputOutputTemplate(input_format=answer_prompt_mixtral)},
                "granite": {"assessment": InputOutputTemplate(input_format=assessment_prompt_granite), "summarization": InputOutputTemplate(input_format=summarization_prompt_granite), "answer" : InputOutputTemplate(input_format=answer_prompt_granite)},
                "llama_8b": {"assessment": InputOutputTemplate(input_format=assessment_prompt_llama), "summarization": InputOutputTemplate(input_format=summarization_prompt_llama), "answer" : InputOutputTemplate(input_format=answer_prompt_llama)},
                "llama_70b": {"assessment": InputOutputTemplate(input_format=assessment_prompt_llama), "summarization": InputOutputTemplate(input_format=summarization_prompt_llama), "answer" : InputOutputTemplate(input_format=answer_prompt_llama)},
                "gpt": {"assessment": InputOutputTemplate(input_format=assessment_prompt_gpt), "summarization": InputOutputTemplate(input_format=summarization_prompt_gpt), "answer" : InputOutputTemplate(input_format=answer_prompt_gpt)},
                "prometheus": {"assessment": InputOutputTemplate(input_format=assessment_prompt_prometheus), "summarization": None, "answer" : None}
                }
