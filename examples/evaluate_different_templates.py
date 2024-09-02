import os
import tempfile

import pandas as pd
from unitxt import add_to_catalog, get_logger, register_local_catalog
from unitxt.api import evaluate, load_dataset
from unitxt.inference_engine import IbmGenAiInferenceEngine
from unitxt.templates import InputOutputTemplate
from unitxt.text_utils import print_dict

logger = get_logger()


# Register a local catalog
def create_path_and_register_as_local_catalog(path):
    if not os.path.exists(path):
        os.mkdir(path)
    register_local_catalog(path)
    return path


catalog_dir = tempfile.gettempdir()  # You can replace with any fixed directory
my_catalog = create_path_and_register_as_local_catalog(catalog_dir)


# Add two templates for entailment tasks to local catalog:
# One template embeds the hypothesis and premise into a single sentence question
# The other templates, places the hypothesis and premise in separate fields with a field prefix.
template1 = InputOutputTemplate(
    input_format='Is "{text_b}" entailed by, neutral to, or contradicts "{text_a}". Answer with one of these following options: {classes}.',
    output_format="{label}",
    postprocessors=[
        "processors.take_first_non_empty_line",
        "processors.lower_case_till_punc",
    ],
)
add_to_catalog(
    template1,
    "templates.my_entailment_as_question",
    catalog_path=my_catalog,
    overwrite=True,
)

template2 = InputOutputTemplate(
    instruction="Indicate whether each hypothesis is entailed by, neutral to, or contradicts the premise. Answer with one of these following options: {classes}.",
    input_format="Premise:\n{text_a}\nHypothesis:\n{text_b}\nEntailment:\n",
    output_format="{label}",
    postprocessors=[
        "processors.take_first_non_empty_line",
        "processors.lower_case_till_punc",
    ],
)
add_to_catalog(
    template2,
    "templates.my_entailment_as_fields",
    catalog_path=my_catalog,
    overwrite=True,
)

# Run inference on mnli (entailment task) on the two templates with both 0 and 3 shot in context learning.
card = "cards.mnli"
model_name = "google/flan-t5-xxl"
inference_model = IbmGenAiInferenceEngine(model_name=model_name, max_new_tokens=32)


df = pd.DataFrame(columns=["template", "num_demos", "f1_micro", "ci_low", "ci_high"])

for template in [
    "templates.my_entailment_as_question",
    "templates.my_entailment_as_fields",
]:
    for num_demos in [0, 3]:
        dataset = load_dataset(
            card=card,
            template=template,
            num_demos=num_demos,
            demos_pool_size=100,
            loader_limit=500,
            max_test_instances=300,
        )

        test_dataset = dataset["test"]

        predictions = inference_model.infer(test_dataset)
        evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

        logger.info(
            f"Sample input and output for template '{template}' and num_demos '{num_demos}':"
        )
        print_dict(
            evaluated_dataset[0],
            keys_to_print=["source", "prediction", "processed_prediction"],
        )
        global_scores = evaluated_dataset[0]["score"]["global"]
        print_dict(
            global_scores,
            keys_to_print=["score_name", "score", "score_ci_low", "score_ci_high"],
        )
        df.loc[len(df)] = [
            template,
            num_demos,
            global_scores["score"],
            global_scores["score_ci_low"],
            global_scores["score_ci_high"],
        ]

df = df.round(decimals=2)
logger.info(df.to_markdown())
