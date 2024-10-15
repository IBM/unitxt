from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.blocks import TaskCard
from unitxt.inference import WMLInferenceEngine
from unitxt.loaders import LoadFromDictionary
from unitxt.templates import TemplatesDict
from unitxt.text_utils import print_dict

logger = get_logger()

# some input rag examples
test_examples = [
    {
        "question": "What foundation models are available in watsonx.ai ?",
        "contexts": [
            "Supported foundation models available with watsonx.ai. Watsonx.ai offers numerous foundation models."
        ],
        "contexts_ids": [0],
        "reference_answers": ["Many Large Language Models are supported by Watsonx.ai"],
    },
    {
        "question": "What foundation models are available in watsonx.ai ?",
        "contexts_ids": [0],
        "contexts": [
            "Supported foundation models available with Meta. Meta AI offers numerous foundation models."
        ],
        "reference_answers": ["Many Large Language Models are supported by Watsonx.ai"],
    },
]

# select the desired metrics.
# Each metric measures a certain aspect of the generated answer (e.g. correctness, faithfulness)
# all available metrics are under "catalog.metrics.llm_as_judge.binary"
# the ones with extension "logprobs" provide a real value between 0 and 1 prediction, the one without it
# provide a binary prediction
metric_names = [
    "answer_correctness_q_a_gt_loose_logprobs",
    "faithfulness_q_c_a_logprobs",
]
metrics_path = "metrics.llm_as_judge.binary"

# select the desired inference model.
# all available models are under "catalog.engines.classification"
model_names = ["llama_3_1_70b_instruct_wml"]
models_path = "engines.classification"


card = TaskCard(
    # Load the data from the dictionary.
    loader=LoadFromDictionary(
        data={"test": test_examples}, data_classification_policy=["public"]
    ),
    # define generation task and templates
    task="tasks.rag.response_generation",
    templates=TemplatesDict(
        {
            "please_respond": "templates.rag.response_generation.please_respond",
            "answer_based_on_context": "templates.rag.response_generation.answer_based_on_context",
        }
    ),
)

# Now let's construct our judge metrics.
# The binary rag judges tasks expect the input fields among the following: "question", "contexts", "ground_truths".
# In our generation task the ground truth are in the "reference_answers" field, so we need to inform the metric
# about this mapping. This is done using the "judge_to_generator_fields_mapping" attribute:
mapping_override = "judge_to_generator_fields_mapping={ground_truths=reference_answers}"

# generate the metrics:
judge_metrics = []
for metric_name in metric_names:
    full_metric_name = f"{metrics_path}.{metric_name}"
    for model_name in model_names:
        full_model_name = f"{models_path}.{model_name}"
        # override the metric with the inference model and mapping
        judge_metrics.append(
            f"{full_metric_name}[inference_model={full_model_name}, {mapping_override}]"
        )

# Verbalize the dataset using the template
dataset = load_dataset(
    card=card, template_card_index="answer_based_on_context", metrics=judge_metrics
)
test_dataset = dataset["test"]

# Infer using flan t5 xl using wml
model_name = "google/flan-t5-xl"
inference_model = WMLInferenceEngine(model_name=model_name, max_new_tokens=32)
predictions = inference_model.infer(test_dataset)
# Evaluate the generated predictions using the selected metrics
evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

# Print results
for instance in evaluated_dataset:
    print_dict(
        instance,
        keys_to_print=[
            "source",
            "prediction",
            "processed_prediction",
            "references",
            "score",
        ],
    )
