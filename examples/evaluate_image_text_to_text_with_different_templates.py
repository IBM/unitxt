from unitxt import settings
from unitxt.api import evaluate
from unitxt.benchmark import Benchmark
from unitxt.inference import (
    LMMSEvalInferenceEngine,
)
from unitxt.logging_utils import get_logger
from unitxt.standard import DatasetRecipe

logger = get_logger()

with settings.context(
    disable_hf_datasets_cache=False,
):
    card = "cards.seed_bench"

    dataset = Benchmark(
        subsets={
            "capitals": DatasetRecipe(
                card=card,
                template="templates.qa.multiple_choice.with_context.lmms_eval[enumerator=capitals]",
                loader_limit=20,
            ),
            "lowercase": DatasetRecipe(
                card=card,
                template="templates.qa.multiple_choice.with_context.lmms_eval[enumerator=lowercase]",
                loader_limit=20,
            ),
            "capitals-greyscale": DatasetRecipe(
                card=card,
                template="templates.qa.multiple_choice.with_context.lmms_eval[enumerator=capitals]",
                loader_limit=20,
                augmentor="augmentors.image.grid_lines",
            ),
        },
    )

    data = list(dataset()["test"])

    model = LMMSEvalInferenceEngine(
        model_type="llava_onevision",
        model_args={"pretrained": "lmms-lab/llava-onevision-qwen2-7b-ov"},
        max_new_tokens=2,
    )

    predictions = model(data)
    results = evaluate(predictions=predictions, data=data)

    for subset in dataset.subsets:
        logger.info(
            f'{subset.title()}: {results[0]["score"]["subsets"][subset]["score"]}'
        )
