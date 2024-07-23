from unitxt.catalog import add_to_catalog
from unitxt.templates import InputOutputTemplate

add_to_catalog(
    InputOutputTemplate(
        instruction="Please act as an impartial judge and evaluate the quality of the text generated"
        " by an AI assistant to the table input given below. Your evaluation should consider"
        " correctness and helpfulness. You will be given a reference text and the assistant generated text."
        " Begin your evaluation by comparing the assistant generated text with the reference text."
        " Identify and correct any mistakes. Be as objective as possible. After providing your explanation,"
        " you must rate the generated text on a scale of 1 to 10 by strictly following this format:"
        ' "[[rating]]", for example: "Rating: [[5]]".\n\n',
        input_format="[Table]\n{question}\n\n"
        "[The Start of Reference Text]\n{reference_answer}\n[The End of Reference Text]\n\n"
        "[The Start of Assistant's Generated Text]\n{answer}\n[The End of Assistant's Generated Text]",
        output_format="[[{rating}]]",
        postprocessors=[
            r"processors.extract_mt_bench_rating_judgment",
        ],
    ),
    "templates.response_assessment.rating.table2text_single_turn_with_reference",
    overwrite=True,
)
