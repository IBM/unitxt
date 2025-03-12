from .templates import InputOutputTemplate

direct_template_dict = {
    "assessment": InputOutputTemplate(
        input_format="""
You are presented with a {response_variable_name} generated subject to a context.
The context includes information relevant to the nature or generation of the {response_variable_name}.
You will assess the quality of the {response_variable_name} subject to an evaluation criteria.
###Context:
{context_variables}

###{response_variable_name_title}:
{response}

###Evaluation criteria:
{criteria_description}
{display_options_instruction}

Briefly assess the quality of the {response_variable_name} subject to the evaluation criteria.
Focus on the evaluation criteria during assessment, do not provide a general assessment.
Assessment:

Lets think step by step """
    ),
    "summarization": InputOutputTemplate(
        input_format="""Transform the following assessment into a concise summary that focuses on the key details, excluding references to the assessment itself.

Assessment: {assessment}
Summary:"""
    ),
    "answer": InputOutputTemplate(
        input_format="""Now consider the evaluation criteria and choose a final answer. Only include the chosen answer in the {response_variable_name}.
###Evaluation criteria:
{criteria_description}
{score_option_instruction}
The selected answer is: """,
        postprocessors=["processors.match_closest_option"],
    ),
}


pairwise_template_dict = {
    "assessment": InputOutputTemplate(
        input_format="""You are provided a pair of {response_variable_name}s ({response_variable_name_title} {option_a} and {response_variable_name_title} {option_b}) generated subject to a context.
You will choose the better quality {response_variable_name} subject to the evaluation criteria.

This is the context:
{context_variables}

This is the evaluation criteria:
{criteria_name}
{criteria_description}

{response_variable_name_title} {option_a}:
{response_a}
{response_variable_name_title} {option_b}:
{response_b}

Keeping the evaluation criteria in mind, briefly assess which {response_variable_name} is better.
Focus on the evaluation criteria during assessment, do not provide a general assessment.
Assessment:

Lets think step by step """
    ),
    "summarization": InputOutputTemplate(
        input_format="""Transform the following assessment into a concise summary that focuses on the key details, excluding references to the assessment itself. The summary must clearly state which {response_variable_name} won.

Assessment: {assessment}
Summary:"""
    ),
    "answer": InputOutputTemplate(
        input_format="""Now considering the evaluation criteria, which {response_variable_name} is better quality? Only include the chosen {response_variable_name}.
{score_option_instruction}
Answer: """,
        postprocessors=["processors.match_closest_option"],
    ),
}
