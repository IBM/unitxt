from src.unitxt import artifact, logging_utils
from src.unitxt.settings_utils import get_settings

logging_utils.enable_explicit_format()
artifact.enable_artifact_caching = False

settings = get_settings()
settings.allow_unverified_code = True
