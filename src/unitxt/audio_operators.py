from typing import Any, Dict, Optional

from .operators import InstanceFieldOperator
from .types import Audio


class ToAudio(InstanceFieldOperator):
    def prepare(self):
        super().prepare()

    def process_instance_value(self, value: Any, instance: Dict[str, Any]) -> Audio:
        return {
            "audio": value,
        }


def audio_to_base64(audio_data: Audio):
    """Convert a HuggingFace Audio instance to a base64-encoded WAV string.

    Args:
        audio_data (dict): The Audio instance from HuggingFace datasets
            Contains 'array', 'sampling_rate', and 'path' keys

    Returns:
        str: Base64-encoded WAV audio
    """
    import base64
    from io import BytesIO

    import soundfile as sf

    # Create a BytesIO buffer to hold the WAV data
    buffer = BytesIO()
    audio = audio_data["audio"]
    # Write the audio array to the buffer in WAV format
    sf.write(buffer, audio["array"], audio["sampling_rate"], format="wav")

    # Get the bytes from the buffer
    wav_bytes = buffer.getvalue()

    # Encode to base64
    return base64.b64encode(wav_bytes).decode("utf-8")


def base64_to_audio(base64_string, sampling_rate: Optional[int] = None):
    import base64

    from datasets import Audio

    audio_bytes = base64.b64decode(base64_string)
    audio_feature = Audio(sampling_rate=sampling_rate)
    return audio_feature.decode_example({"bytes": audio_bytes, "path": None})
