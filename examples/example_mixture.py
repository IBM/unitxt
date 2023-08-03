from src.unitxt import add_to_catalog, load_dataset
from src.unitxt.blocks import LoadHF
from src.unitxt.common import CommonRecipe
from src.unitxt.fusion import WeightedFusion
from src.unitxt.text_utils import print_dict

fusion = WeightedFusion(
    origins=[
        "recipes.wnli_3_shot",
        "recipes.wnli_3_shot",
    ],
    weights=[1, 1],
    total_examples=4,
)

add_to_catalog(fusion, "benchmarks.glue", overwrite=True)

dataset = load_dataset("benchmarks.glue")

for example in dataset["train"]:
    print_dict(example)
    print()
