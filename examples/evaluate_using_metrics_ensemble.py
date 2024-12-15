from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.inference import (
    HFPipelineBasedInferenceEngine,
)
from unitxt.metrics import MetricsEnsemble

logger = get_logger()

# define the metrics ensemble
ensemble_metric = MetricsEnsemble(
    metrics=[
        "metrics.llm_as_judge.rating.llama_3_70b_instruct.generic_single_turn",
        "metrics.llm_as_judge.rating.llama_3_8b_instruct_ibm_genai_template_mt_bench_single_turn",
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
    model_name="Qwen/Qwen1.5-0.5B-Chat", max_new_tokens=32
)
# Change to this to infer with external APIs:
# CrossProviderInferenceEngine(model="llama-3-2-1b-instruct", provider="watsonx")
# The provider can be one of: ["watsonx", "together-ai", "open-ai", "aws", "ollama", "bam"]

predictions = engine.infer(dataset)

# Evaluate the predictions using the defined metric.
results = evaluate(predictions=predictions, data=dataset)

# Print results
print(
    results.instance_scores.to_df(
        columns=[
            "question",
            "prediction",
            "processed_prediction",
            "references",
            "ensemble_0_llama_3_70b_instruct_template_generic_single_turn",
            "ensemble_1_llama_3_8b_instruct_ibm_genai_template_mt_bench_single_turn",
            "score",
            "score_name",
        ],
    )
)
