from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.blocks import TaskCard
from unitxt.inference import WMLInferenceEngine
from unitxt.loaders import LoadFromDictionary
from unitxt.templates import TemplatesDict

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

if __name__ == "__main__":
    # define our card for the generation task
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

    # Select the desired metric(s).
    # Each metric measures a certain aspect of the generated answer (answer_correctness, faithfulness,
    # answer_relevance, context_relevance and correctness_holistic).
    # All available metrics are under "catalog.metrics.rag"
    # Those with extension "logprobs" provide a real value prediction in [0,1], the others provide a binary prediction.
    # By default, all judges use llama_3_1_70b_instruct_wml. We will soon see how to change this.
    metric_name = "metrics.rag.answer_correctness.llama_3_1_70b_instruct_wml_q_a_gt_loose_logprobs"

    # The binary rag judges tasks expect the input fields among the following: "question", "contexts", "ground_truths".
    # In our generation task the ground truth are in the "reference_answers" field, so we need to inform the metric
    # about this mapping. This is done using the "judge_to_generator_fields_mapping" attribute:
    mapping_override = (
        "judge_to_generator_fields_mapping={ground_truths=reference_answers}"
    )
    correctness_judge_metric_llama = f"{metric_name}[{mapping_override}]"

    # We can also use another inference model by overriding the "model" attribute of the metric.
    # all available models for this judge are under "catalog.engines.classification"
    mixtral_engine = "engines.classification.mixtral_8x7b_instruct_v01_wml"
    correctness_judge_metric_mixtral = (
        f"{metric_name}[{mapping_override}, inference_model={mixtral_engine}]"
    )

    metrics = [correctness_judge_metric_llama, correctness_judge_metric_mixtral]

    # Verbalize the dataset using the template
    dataset = load_dataset(
        card=card, template_card_index="answer_based_on_context", metrics=metrics
    )
    test_dataset = dataset["test"]

    # Infer using flan t5 xl using wml
    model_name = "google/flan-t5-xl"
    model = WMLInferenceEngine(model_name=model_name, max_new_tokens=32)
    predictions = model(test_dataset)

    # Evaluate the generated predictions using the selected metrics
    results = evaluate(predictions=predictions, data=test_dataset)

    print("Global Results:")
    print(results.global_scores.summary)

    print("Instance Results:")
    print(results.instance_scores.summary)
