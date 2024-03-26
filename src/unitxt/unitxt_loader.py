from unitxt import dataset

from .loaders import Loader
from .standard import StandardRecipe
from .stream import MultiStream


class LoadUnitxt(Loader):
    dataset: str

    def process(self):

        try:
            dataset_params = dataset.parse(self.dataset)
            recipe = StandardRecipe(**dataset_params)
        except TypeError as e:
            raise TypeError(
                f"Error loading card with params {dataset_params} due to: \n\n*****\n{e}\n\nHelp: {StandardRecipe.__doc__}"
            ) from e
        stream = MultiStream(recipe())
        return stream
