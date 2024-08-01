from unitxt import add_to_catalog
from unitxt.templates import InputOutputTemplate

# Use the same processor of MT Bench to extract the result
add_to_catalog(
    InputOutputTemplate(
        instruction="A task was given to a model with a list of options as possible answers,"
        " there's also the golden response. Act as a judge comparing the model's response"
        " and the golden response and give a rating of [[10]] if the model's response matches"
        " the golden response otherwise provide a rating of [[0]].\n\n"
        "Strictly follow the output format below and maintain the double square brackets.\n"
        "```\nrating: [[your-rating]]\n```\n\n"
        "The task, the model's response chosen from the options list, and the golden response are as follows:\n",
        input_format="Task: {question}\n\n"
        "Model's Response: {answer}\n\n"
        "Golden Response: {reference_answer}\n\n",
        output_format="[[{rating}]]",
        postprocessors=[
            r"processors.extract_mt_bench_rating_judgment",
        ],
    ),
    "templates.response_assessment.rating.mixeval_multi_choice_parser",
    overwrite=True,
)
