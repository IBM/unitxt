from json.decoder import JSONDecodeError

from .artifact import Artifact, UnitxtArtifactNotFoundError, fetch_artifact
from .logging_utils import get_logger
from .parsing_utils import parse_key_equals_value_string_to_dict
from .register import _reset_env_local_catalogs, register_all_artifacts
from .settings_utils import get_settings
from .standard import BaseRecipe

logger = get_logger()
settings = get_settings()


def fetch(artifact_name):
    try:
        artifact, _ = fetch_artifact(artifact_name)
        return artifact
    except (UnitxtArtifactNotFoundError, JSONDecodeError):
        return None


def parse(query: str):
    return parse_key_equals_value_string_to_dict(query)


def get_dataset_artifact(dataset):
    if isinstance(dataset, BaseRecipe):
        return dataset
    assert isinstance(
        dataset, str
    ), "dataset should be string description of recipe, or recipe object."
    _reset_env_local_catalogs()
    register_all_artifacts()
    recipe = fetch(dataset)
    if recipe is None:
        args = parse(dataset)
        if "__type__" not in args:
            args["__type__"] = settings.default_recipe
        recipe = Artifact.from_dict(args)
    return recipe
