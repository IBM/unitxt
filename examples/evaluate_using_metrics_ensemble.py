from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.inference import CrossProviderInferenceEngine
from unitxt.metrics import MetricsEnsemble

logger = get_logger()

# define the metrics ensemble
ensemble_metric = MetricsEnsemble(
    metrics=[
        "metrics.llm_as_judge.direct.watsonx.llama3_3_70b[criteria=metrics.llm_as_judge.direct.criteria.answer_relevance, context_fields=[question]]",
        "metrics.llm_as_judge.direct.watsonx.llama3_3_70b[criteria=metrics.llm_as_judge.direct.criteria.correctness_based_on_ground_truth, context_fields=[question,answers]]",
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

# Change to this to infer with external APIs:
model = CrossProviderInferenceEngine(model="llama-3-2-1b-instruct", provider="watsonx")

predictions = model(dataset)

# Evaluate the predictions using the defined metric.
results = evaluate(predictions=predictions, data=dataset)

print("Global Results:")
print(results.global_scores.summary)

print("Instance Results:")
print(results.instance_scores.summary)
