from .artifact import Artifact, UnitxtArtifactNotFoundError, fetch_artifact
from .logging_utils import get_logger
from .register import _reset_env_local_catalogs, register_all_artifacts
from .settings_utils import get_settings

logger = get_logger()
settings = get_settings()


def fetch(artifact_name):
    try:
        artifact, _ = fetch_artifact(artifact_name)
        return artifact
    except UnitxtArtifactNotFoundError:
        return None


def parse(query: str):
    """Parses a query of the form 'key1=value1,key2=value2,...' into a dictionary."""
    result = {}
    kvs = query.split(",")
    if len(kvs) == 0:
        raise ValueError(
            'Illegal query: "{query}" should contain at least one assignment of the form: key1=value1,key2=value2'
        )
    for kv in kvs:
        key_val = kv.split("=")
        if (
            len(key_val) != 2
            or len(key_val[0].strip()) == 0
            or len(key_val[1].strip()) == 0
        ):
            raise ValueError(
                f'Illegal query: "{query}" with wrong assignment "{kv}" should be of the form: key=value.'
            )
        key, val = key_val
        if val.isdigit():
            result[key] = int(val)
        elif val.replace(".", "", 1).isdigit():
            result[key] = float(val)
        else:
            result[key] = val

    return result


def get_dataset_artifact(dataset_str):
    _reset_env_local_catalogs()
    register_all_artifacts()
    recipe = fetch(dataset_str)
    if recipe is None:
        args = parse(dataset_str)
        if "type" not in args:
            args["type"] = settings.default_recipe
        recipe = Artifact.from_dict(args)
    return recipe
