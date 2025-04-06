from unitxt.api import create_dataset, evaluate

classes = ["positive", "negative"]

dataset = [
    {"text": "I am happy.", "label": "positive", "classes": classes},
    {"text": "It was a great movie.", "label": "positive", "classes": classes},
    {"text": "I never felt so bad", "label": "negative", "classes": classes},
]

predictions = ["Positive.", "negative.", "negative"]

dataset = create_dataset(
    task="tasks.classification.multi_class",
    format="formats.chat_api",
    test_set=dataset,
    postprocessors=["processors.take_first_word", "processors.lower_case"],
)

results = evaluate(predictions, dataset["test"])

# Print Results:

print(f"Final Score ({results.global_scores.score_name}):")
print(results.global_scores.score)

print("Global Results:")
print(results.global_scores.summary)

print("Instance Results:")
print(results.instance_scores.summary)
