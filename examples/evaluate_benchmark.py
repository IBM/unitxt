from unitxt.api import evaluate
from unitxt.benchmark import Benchmark
from unitxt.inference import (
    CrossProviderInferenceEngine,
)
from unitxt.standard import StandardRecipe

benchmark = Benchmark(
    format="formats.user_agent",
    max_samples_per_subset=5,
    loader_limit=300,
    subsets={
        "cola": StandardRecipe(
            card="cards.cola",
            template="templates.classification.multi_class.instruction",
        ),
        "mnli": StandardRecipe(
            card="cards.mnli",
            template="templates.classification.multi_class.relation.default",
        ),
        "mrpc": StandardRecipe(
            card="cards.mrpc",
            template="templates.classification.multi_class.relation.default",
        ),
        "qnli": StandardRecipe(
            card="cards.qnli",
            template="templates.classification.multi_class.relation.default",
        ),
        "rte": StandardRecipe(
            card="cards.rte",
            template="templates.classification.multi_class.relation.default",
        ),
        "sst2": StandardRecipe(
            card="cards.sst2", template="templates.classification.multi_class.title"
        ),
        "stsb": StandardRecipe(
            card="cards.stsb", template="templates.regression.two_texts.title"
        ),
        "wnli": StandardRecipe(
            card="cards.wnli",
            template="templates.classification.multi_class.relation.default",
        ),
    },
)

test_dataset = list(benchmark()["test"])


# Infere using llama-3-2-1b base using Watsonx API
model = CrossProviderInferenceEngine(model="llama-3-2-1b-instruct", provider="watsonx")
"""
We are using a CrossProviderInferenceEngine inference engine that supply api access to provider such as:
watsonx, bam, openai, azure, aws and more.

For the arguments these inference engines can receive, please refer to the classes documentation or read
about the the open ai api arguments the CrossProviderInferenceEngine follows.
"""

predictions = model(test_dataset)
results = evaluate(predictions=predictions, data=test_dataset)

print(
    results.subsets_scores,
)
