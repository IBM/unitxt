from src.unitxt import add_to_catalog
from src.unitxt.templates import InputOutputTemplate

add_to_catalog(
    InputOutputTemplate(
        instruction="""You are given a text. Does this text contain any grammatical errors or spelling mistakes? Answer only "Yes" or "No".\n""",
        input_format="Text: {text}",
        output_format="{label}",
        target_prefix="Answer: ",
        postprocessors=[
            "processors.take_first_word",
            "processors.lower_case",
            "processors.yes_no_to_int",
        ],
    ),
    "templates.grammatical_error_detection.yes_no",
    overwrite=True,
)
