from json.decoder import JSONDecodeError
from typing import Any, Dict, Optional

from .artifact import Artifact, UnitxtArtifactNotFoundError, fetch_artifact
from .logging_utils import get_logger
from .parsing_utils import parse_key_equals_value_string_to_dict
from .register import _reset_env_local_catalogs, register_all_artifacts
from .settings_utils import get_settings
from .standard import DatasetRecipe

logger = get_logger()
settings = get_settings()


def fetch(artifact_name: str, overwrite_kwargs: Optional[Dict[str, Any]] = None):
    try:
        artifact, _ = fetch_artifact(artifact_name, overwrite_kwargs=overwrite_kwargs)
        return artifact
    except (UnitxtArtifactNotFoundError, JSONDecodeError):
        return None


def parse(query: str) -> dict:
    return parse_key_equals_value_string_to_dict(query)


def get_dataset_artifact(dataset, overwrite_kwargs: Optional[Dict[str, Any]] = None):
    if isinstance(dataset, DatasetRecipe):
        return dataset
    assert isinstance(
        dataset, str
    ), "dataset should be string description of recipe, or recipe object."
    _reset_env_local_catalogs()
    register_all_artifacts()
    recipe = fetch(dataset, overwrite_kwargs=overwrite_kwargs)
    if recipe is None:
        args = parse(dataset)
        if "__type__" not in args:
            args["__type__"] = settings.default_recipe
        if overwrite_kwargs is not None:
            args.update(overwrite_kwargs)
        recipe = Artifact.from_dict(args)
    return recipe
