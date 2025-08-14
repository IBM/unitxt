from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog
from unitxt.templates import InputOutputTemplate
from unitxt.types import Audio

add_to_catalog(
    Task(
        input_fields={
            "audio": Audio,
            "target_language": str,
        },
        reference_fields={"translation": str},
        prediction_type=str,
        metrics=["metrics.normalized_sacrebleu"],
        default_template=InputOutputTemplate(
            # input_format="{audio}listen to the speech and translate it to {target_language}",
            input_format="{audio}translate the speech to {target_language}",
            output_format="{translation}",
            postprocessors=["processors.normalize_text_basic_with_whisper"],
        ),
    ),
    "tasks.translation.speech",
    overwrite=True,
)
