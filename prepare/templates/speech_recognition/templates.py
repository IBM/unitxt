from unitxt.catalog import add_to_catalog
from unitxt.templates import InputOutputTemplate

add_to_catalog(
    InputOutputTemplate(
        input_format="{audio}can you transcribe the speech into a written format?",
        output_format="{text}",
        postprocessors=["processors.normalize_text_with_whisper"],
    ),
    "templates.speech_recognition.default",
    overwrite=True,
)
