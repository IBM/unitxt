from .artifact import Artifact
from . import blocks

import inspect


def register_blocks():
    # Iterate over every object in the blocks module
    for name, obj in inspect.getmembers(blocks):
        # Make sure the object is a class
        if inspect.isclass(obj):
            # Make sure the class is a subclass of Artifact (but not Artifact itself)
            if issubclass(obj, Artifact) and obj is not Artifact:
                Artifact.register_class(obj)


register_blocks()
