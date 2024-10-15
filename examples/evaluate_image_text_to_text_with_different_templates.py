from tqdm import tqdm
from unitxt import settings
from unitxt.api import evaluate
from unitxt.benchmark import Benchmark
from unitxt.inference import (
    HFLlavaInferenceEngine,
)
from unitxt.standard import StandardRecipe
from unitxt.templates import MultipleChoiceTemplate
from unitxt.text_utils import print_dict

with settings.context(
    disable_hf_datasets_cache=False,
):
    inference_model = HFLlavaInferenceEngine(
        model_name="llava-hf/llava-interleave-qwen-0.5b-hf", max_new_tokens=32
    )

    dataset = Benchmark(
        format="formats.models.llava_interleave",
        subsets={
            "capitals": StandardRecipe(
                card="cards.ai2d",
                template="templates.qa.multiple_choice.with_context.title",
                format="formats.models.llava_interleave",
                loader_limit=20,
            ),
            "lowercase": StandardRecipe(
                card="cards.ai2d",
                template="templates.qa.multiple_choice.with_context.title[enumerator=lowercase]",
                format="formats.models.llava_interleave",
                loader_limit=20,
            ),
            "numbers": StandardRecipe(
                card="cards.ai2d",
                template=MultipleChoiceTemplate(
                    instruction="Answer the multiple choice Question from one of the Choices (choose from {numerals}) based on the {context_type}.",
                    input_format="{context_type}:\n{context}\nQuestion:\n{question}\nChoices:\n{choices}",
                    target_prefix="Answer:\n",
                    target_field="answer",
                    choices_separator="\n",
                    enumerator="numbers",
                    postprocessors=[
                        "processors.to_string_stripped",
                        "processors.first_character",
                    ],
                    title_fields=["context_type"],
                ),
                format="formats.models.llava_interleave",
                loader_limit=20,
            ),
        },
    )()

    test_dataset = list(tqdm(dataset["test"], total=60))

    predictions = inference_model.infer(test_dataset)
    evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

    print_dict(
        evaluated_dataset[0],
        keys_to_print=[
            "source",
            "media",
            "references",
            "processed_prediction",
            "score",
        ],
    )
