from unitxt import add_to_catalog
from unitxt.templates import InputOutputTemplate

add_to_catalog(
    InputOutputTemplate(
        instruction="Please act as an impartial judge and evaluate the quality of the response provided"
        " by an AI assistant to the user input displayed below. Your evaluation should consider"
        " factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of"
        " detail of the response. Begin your evaluation by providing a short explanation. Be as"
        " objective as possible. After providing your explanation, you must rate the response"
        ' on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example:'
        ' "Rating: [[5]]".\n\n',
        input_format="[User input]\n{question}\n\n"
        "[Assistant's respond]\n{answer}\n[The End of Assistant's respond]",
        output_format="[[{rating}]]",
        postprocessors=[
            r"processors.extract_mt_bench_rating_judgment",
        ],
    ),
    "templates.response_assessment.rating.generic_single_turn",
    overwrite=True,
)
