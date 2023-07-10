###############
# `ls -1 src/unitxt | grep '\.py$' | grep -Ev 'dataset\.py|__init__\.py' | sort`:
# artifact.py
# blocks.py
# card.py
# catalog.py
# collections.py
# common.py
# file_utils.py
# fusion.py
# generator_utils.py
# instructions.py
# loaders.py
# load.py
# metric.py
# metrics.py
# normalizers.py
# operator.py
# operators.py
# processors.py
# recipe.py
# register.py
# splitters.py
# split_utils.py
# stream.py
# task.py
# templates.py
# text_utils.py
# utilize.py
# validate.py
#####
# imports for hf system:
#####
from .artifact import __file__ as _
from .blocks import __file__ as _
from .card import __file__ as _
from .catalog import __file__ as _
from .collections import __file__ as _
from .common import __file__ as _
from .file_utils import __file__ as _

# from .fusion import __file__
from .generator_utils import __file__ as _
from .instructions import __file__ as _
from .loaders import __file__ as _
from .load import __file__ as _
from .metric import __file__ as _
from .metrics import __file__ as _
from .normalizers import __file__ as _
from .operator import __file__ as _
from .operators import __file__ as _
from .processors import __file__ as _
from .recipe import __file__ as _
from .register import __file__ as _
from .schema import __file__ as _
from .splitters import __file__ as _
from .split_utils import __file__ as _
from .stream import __file__ as _
from .task import __file__ as _
from .templates import __file__ as _
from .text_utils import __file__ as _

# from .utilize import __file__ as _
# from .validate import __file__ as _
#############

from .register import register_blocks
from .artifact import Artifact

import datasets


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


class Unitext(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.1")
    builder_configs = {}

    @property
    def generators(self):
        register_blocks()
        if not hasattr(self, "_generators") or self._generators is None:
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
