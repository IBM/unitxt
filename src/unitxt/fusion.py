from typing import List, Optional

from .loaders import Loader
from .splitters import Splitter
from .stream import MultiStream
from .task import Tasker

# class Fusion(StreamSource):
#     pass


# class RecipeFusion(StreamSource):
#     recepies: List[Recipe]

#     def __call__(self) -> MultiStream:
#         for recipe in self.recepies:
#             stream = recipe()
#         return stream
