from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.inference import (
    HFPipelineBasedInferenceEngine,
)
from unitxt.metrics import MetricsEnsemble
from unitxt.text_utils import print_dict

logger = get_logger()

# define the metrics ensemble
ensemble_metric = MetricsEnsemble(
    metrics=[
        "metrics.llm_as_judge.rating.llama_3_70b_instruct_ibm_genai_template_generic_single_turn",
        "metrics.llm_as_judge.rating.llama_3_70b_instruct_ibm_genai_template_mt_bench_single_turn",
    ],
    weights=[0.75, 0.25],
)
# Use the HF load_dataset API, to load the squad QA dataset using the standard template in the catalog.
# We set loader_limit to 20 to reduce download time.
dataset = load_dataset(
    card="cards.squad",
    template="templates.qa.with_context.simple",
    format="formats.chat_api",
    metrics=[ensemble_metric],
    loader_limit=20,
    max_test_instances=10,
    split="test",
)

# Infer using Llama-3.2-1B base using HF API
engine = HFPipelineBasedInferenceEngine(
    model_name="meta-llama/Llama-3.2-1B", max_new_tokens=32
)
# Change to this to infer with external APIs:
# CrossProviderInferenceEngine(model="llama-3-2-1b-instruct", provider="watsonx")
# The provider can be one of: ["watsonx", "together-ai", "open-ai", "aws", "ollama", "bam"]

predictions = engine.infer(dataset)

# Evaluate the predictions using the defined metric.
evaluated_dataset = evaluate(predictions=predictions, data=dataset)

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
