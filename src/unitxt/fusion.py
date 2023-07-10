from .stream import MultiStream

from .loaders import Loader
from .splitters import Splitter
from .task import Tasker
from .render import Templater

from typing import Optional, List


class Fusion(StreamSource):
    pass


class RecipeFusion(StreamSource):
    recepies: List[Recipe]

    def __call__(self) -> MultiStream:
        for recipe in self.recepies:
            stream = recipe()
        return stream
