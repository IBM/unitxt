from unitxt import settings
from unitxt.api import evaluate
from unitxt.artifact import fetch_artifact
from unitxt.benchmark import Benchmark
from unitxt.inference import (
    LMMSEvalInferenceEngine,
)
from unitxt.logging_utils import get_logger
from unitxt.standard import StandardRecipe

logger = get_logger()

with settings.context(
    disable_hf_datasets_cache=False,
):
    subsets = {}
    for card in ["cards.seed_bench"]:
        task_card, _ = fetch_artifact(card)
        loader0 = task_card.loader
        for enumerator in ["capitals", "lowercase"]:
            for augmentor in [None, "augmentors.image.white_noise"]:
                recipei = StandardRecipe(
                    card=card,
                    template=f"templates.qa.multiple_choice.with_context.lmms_eval[enumerator={enumerator}]",
                    loader_limit=100,
                    augmentor=augmentor,
                )
                recipei.loading.steps[0] = loader0
                loader0.loader_limit = recipei.loader_limit
                subsets[f"{card} {enumerator} {augmentor}"] = recipei

    benchmark = Benchmark(subsets=subsets)

    data = list(benchmark()["test"])

    inference_model = LMMSEvalInferenceEngine(
        model_type="llava",
        model_args={"pretrained": "llava-hf/llava-v1.6-mistral-7b-hf"},
        max_new_tokens=2,
    )

    predictions = inference_model.infer(data)
    results = evaluate(predictions=predictions, data=data)

    for subset in benchmark.subsets:
        logger.info(
            f'{subset.title()}: {results[0]["score"]["subsets"][subset]["score"]}'
        )
