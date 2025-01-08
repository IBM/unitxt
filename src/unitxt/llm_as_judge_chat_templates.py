from .templates import InputOutputTemplate

direct_template_dict = {
    "assessment": InputOutputTemplate(
        input_format="""
You are presented with a response generated subject to a context.
The context includes information relevant to the nature or generation of the response.
You will assess the quality of the response subject to an evaluation criteria.
###Context:
{context_variables}
###Response:
{response}
###Evaluation criteria:
{criteria_description}
{display_options_instruction}
Briefly assess the quality of the response subject to the evaluation criteria.
Focus on the evaluation criteria during assessment, do not provide a general assessment.
Assessment: """
    ),
    "summarization": InputOutputTemplate(
        input_format="""Transform the following assessment into a concise summary that focuses on the key details, excluding references to the assessment itself.

Assessment: {assessment}
Summary:"""
    ),
    "answer": InputOutputTemplate(
        input_format="""Now consider the evaluation criteria and choose a final answer. Only include the chosen answer in the response.
###Evaluation criteria:
{criteria_description}
{score_option_instruction}
The selected answer is: """,
        postprocessors=["processors.match_closest_option"],
    ),
}


pairwise_template_dict = {
    "assessment": InputOutputTemplate(
        input_format="""You are provided a pair of responses (Response {option_a} and Response {option_b}) generated subject to a context.
You will choose the better quality response subject to the evaluation criteria.

This is the context:
{context_variables}
This is the evaluation criteria:
{criteria_name}
{criteria_description}
Response {option_a}:
{response_a}
Response {option_b}:
{response_b}

Keeping the evaluation criteria in mind, briefly assess which response is better.
Focus on the evaluation criteria during assessment, do not provide a general assessment.
Assessment: """
    ),
    "summarization": InputOutputTemplate(
        input_format="""Transform the following assessment into a concise summary that focuses on the key details, excluding references to the assessment itself. The summary must clearly state which response won.

Assessment: {assessment}
Summary:"""
    ),
    "answer": InputOutputTemplate(
        input_format="""Now considering the evaluation criteria, which response is better quality? Only include the chosen response.
{score_option_instruction}
Answer: """,
        postprocessors=["processors.match_closest_option"],
    ),
}
