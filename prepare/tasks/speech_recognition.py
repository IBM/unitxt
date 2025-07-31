from unitxt.catalog import add_to_catalog
from unitxt.task import Task
from unitxt.templates import InputOutputTemplate
from unitxt.types import Audio

add_to_catalog(
    Task(
        input_fields={
            "audio": Audio,
        },
        reference_fields={"text": str},
        prediction_type=str,
        metrics=["metrics.wer"],
        default_template=InputOutputTemplate(
            input_format="{audio}can you transcribe the speech into a written format?",
            output_format="{text}",
            postprocessors=["processors.normalize_text_with_whisper"],
        ),
    ),
    "tasks.speech_recognition",
    overwrite=True,
)
