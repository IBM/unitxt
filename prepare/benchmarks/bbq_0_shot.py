from unitxt.benchmark import Benchmark
from unitxt.catalog import add_to_catalog
from unitxt.standard import StandardRecipe

subsets = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Race_x_SES",
    "Race_x_gender",
    "Religion",
    "SES",
    "Sexual_orientation",
]
benchmark = Benchmark(
    subsets={
        subset: StandardRecipe(
            card=f"cards.safety.bbq.{subset}",
            template_card_index=2,
            format="formats.chat_api",
            num_demos=0,
            demos_taken_from="test"
        ) for subset in subsets
    },
)

add_to_catalog(benchmark, "benchmarks.bbq_0_shot", overwrite=True)
