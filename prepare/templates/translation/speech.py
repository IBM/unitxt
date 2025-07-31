from unitxt.catalog import add_to_catalog
from unitxt.templates import InputOutputTemplate

add_to_catalog(
    InputOutputTemplate(
        input_format="{audio}listen to the speech and translate it to {target_language}",
        output_format="{translation}",
    ),
    "templates.translation.speech.default",
    overwrite=True,
)
