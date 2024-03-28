from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import OutputQuantizingTemplate

add_to_catalog(
    OutputQuantizingTemplate(
        instruction=(
            "Evaluate the similarity between them and classify them into classes from 0-5 as follows:\n"
            "0 : The two sentences are completely dissimilar.\n"
            "1 : The two sentences are not equivalent, but are on the same topic.\n"
            "2 : The two sentences are not equivalent, but share some details.\n"
            "3 : The two sentences are roughly equivalent, but some important information differs/missing.\n"
            "4 : The two sentences are mostly equivalent, but some unimportant details differ.\n"
            "5 : The two sentences are completely equivalent, as they mean the same thing."
        ),
        input_format="Sentence 1: {text1} Sentence 2: {text2}",
        output_format="{attribute_value}",
        quantum=1,
    ),
    "templates.regression.two_texts.similarity.flan",
    overwrite=True,
)
