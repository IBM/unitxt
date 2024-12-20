from unitxt.api import evaluate
from unitxt.benchmark import Benchmark
from unitxt.inference import (
    CrossProviderInferenceEngine,
)
from unitxt.standard import DatasetRecipe
from unitxt.text_utils import print_dict

benchmark = Benchmark(
    format="formats.user_agent",
    max_samples_per_subset=5,
    loader_limit=30,
    subsets={
        "cola": DatasetRecipe(
            card="cards.cola",
            template="templates.classification.multi_class.instruction",
        ),
        "mnli": DatasetRecipe(
            card="cards.mnli",
            template="templates.classification.multi_class.relation.default",
        ),
        "mrpc": DatasetRecipe(
            card="cards.mrpc",
            template="templates.classification.multi_class.relation.default",
        ),
        "qnli": DatasetRecipe(
            card="cards.qnli",
            template="templates.classification.multi_class.relation.default",
        ),
        "rte": DatasetRecipe(
            card="cards.rte",
            template="templates.classification.multi_class.relation.default",
        ),
        "sst2": DatasetRecipe(
            card="cards.sst2", template="templates.classification.multi_class.title"
        ),
        "stsb": DatasetRecipe(
            card="cards.stsb", template="templates.regression.two_texts.title"
        ),
        "wnli": DatasetRecipe(
            card="cards.wnli",
            template="templates.classification.multi_class.relation.default",
        ),
    },
)

test_dataset = list(benchmark()["test"])


# Infer using llama-3-2-1b base using Watsonx API
model = CrossProviderInferenceEngine(model="llama-3-2-1b-instruct", provider="watsonx")
"""
We are using a CrossProviderInferenceEngine inference engine that supply api access to provider such as:
watsonx, bam, openai, azure, aws and more.

For the arguments these inference engines can receive, please refer to the classes documentation or read
about the the open ai api arguments the CrossProviderInferenceEngine follows.
"""

predictions = model(test_dataset)
results = evaluate(predictions=predictions, data=test_dataset)

print("Global Results:")
print(results.global_scores.summary)

print("Subsets Results:")
print(results.subsets_scores.summary)
