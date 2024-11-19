from unitxt.api import evaluate
from unitxt.benchmark import Benchmark
from unitxt.inference import (
    CrossProviderInferenceEngine,
)
from unitxt.standard import StandardRecipe
from unitxt.text_utils import print_dict

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
inference_model = CrossProviderInferenceEngine(
    model="llama-3-2-1b-instruct", provider="watsonx"
)
"""
We are using a CrossProviderInferenceEngine inference engine that supply api access to provider such as:
watsonx, bam, openai, azure, aws and more.

For the arguments these inference engines can receive, please refer to the classes documentation or read
about the the open ai api arguments the CrossProviderInferenceEngine follows.
"""

predictions = inference_model.infer(test_dataset)
evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

print_dict(
    evaluated_dataset[0],
    keys_to_print=[
        "source",
        "prediction",
        "subset",
    ],
)
print_dict(
    evaluated_dataset[0]["score"]["subsets"],
)
