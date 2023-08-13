import datasets

from .artifact import Artifact, UnitxtArtifactNotFoundError
from .artifact import __file__ as _
from .artifact import fetch_artifact
from .blocks import __file__ as _
from .card import __file__ as _
from .catalog import __file__ as _
from .collections import __file__ as _
from .common import __file__ as _
from .dataclass import __file__ as _
from .file_utils import __file__ as _
from .fusion import __file__ as _
from .generator_utils import __file__ as _
from .instructions import __file__ as _
from .load import __file__ as _
from .loaders import __file__ as _
from .metric import __file__ as _
from .metrics import __file__ as _
from .normalizers import __file__ as _
from .operator import __file__ as _
from .operators import __file__ as _
from .processors import __file__ as _
from .recipe import __file__ as _
from .register import __file__ as _
from .register import register_all_artifacts
from .schema import __file__ as _
from .split_utils import __file__ as _
from .splitters import __file__ as _
from .stream import __file__ as _
from .task import __file__ as _
from .templates import __file__ as _
from .text_utils import __file__ as _
from .utils import __file__ as _
from .validate import __file__ as _
from .type_utils import __file__ as _
from .hf_utils import __file__ as _
from .dict_utils import __file__ as _
from .random_utils import __file__ as _
from .version import __file__ as _

from .version import version


def fetch(artifact_name):
    try:
        artifact, _ = fetch_artifact(artifact_name)
        return artifact
    except UnitxtArtifactNotFoundError:
        return None


def parse(query: str):
    """
    Parses a query of the form 'key1=value1,key2=value2,...' into a dictionary.
    """
    result = {}
    for kv in query.split(","):
        parts = kv.split("=")
        if parts[1].isdigit():
            result[parts[0]] = int(parts[1])
        elif parts[1].replace(".", "", 1).isdigit():
            result[parts[0]] = float(parts[1])

        result[parts[0]] = parts[1]

    return result


class Dataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version(version)
    builder_configs = {}

    @property
    def generators(self):
        register_all_artifacts()
        if not hasattr(self, "_generators") or self._generators is None:
            recipe = fetch(self.config.name)
            if recipe is None:
                args = parse(self.config.name)
                if "type" not in args:
                    args["type"] = "common_recipe"
                recipe = Artifact.from_dict(args)
            self._generators = recipe()
        return self._generators

    def _info(self):
        return datasets.DatasetInfo()

    def _split_generators(self, _):
        return [datasets.SplitGenerator(name=name, gen_kwargs={"split_name": name}) for name in self.generators.keys()]

    def _generate_examples(self, split_name):
        generator = self.generators[split_name]
        for i, row in enumerate(generator):
            yield i, row

    def _download_and_prepare(self, dl_manager, verification_mode, **prepare_splits_kwargs):
        result = super()._download_and_prepare(dl_manager, "no_checks", **prepare_splits_kwargs)
        return result
