from unitxt.catalog import add_to_catalog
from unitxt.templates import InputOutputTemplate

add_to_catalog(
    InputOutputTemplate(
        # input_format="{audio}listen to the speech and translate it to {target_language}",
        input_format="{audio}translate the speech to {target_language}",
        output_format="{translation}",
        postprocessors=["processors.normalize_text_basic_with_whisper"],
    ),
    "templates.translation.speech.default",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        # input_format="{audio}listen to the speech and translate it to {target_language}",
        input_format="{audio}translate the speech to {target_language}",
        output_format="{translation}",
        # postprocessors=["processors.normalize_text_basic_with_whisper"],
    ),
    "templates.translation.speech.no_norm",
    overwrite=True,
)
