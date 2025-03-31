import json

import evaluate
from datasets import load_dataset

# Use the HF load_dataset API, to load the wnli entailment dataset using the standard template in the catalog for relation task with 2-shot in-context learning.
# We set loader_limit to 200 to limit reduce download time.
dataset = load_dataset(
    "unitxt/data",
    "card=cards.wnli,template=templates.classification.multi_class.relation.default,num_demos=2,demos_pool_size=100,loader_limit=200",
    trust_remote_code=True,
    split="test",
)

# Print the resulting dataset.
# The 'source' field contains the input to the model, and the 'references' field contains
# that expected answer.

print("Sample dataset instance:")
print(json.dumps(dataset[0], indent=4))

# Generate predictions which are always entailment. Can be replaced with any inference method.
predictions = ["entailment" for _ in dataset]

# Use the huggingface evaluate API to evaluate using the built in metrics for the task
# (f1_micro, f1_macro, accuracy, including confidence intervals)

metric = evaluate.load("unitxt/metric")
results = metric.compute(predictions=predictions, references=dataset)

print("Global Results:")
print(results.global_scores.summary)
