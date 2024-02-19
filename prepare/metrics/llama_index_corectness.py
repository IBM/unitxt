from unitxt import add_to_catalog
from unitxt.metrics import LlamaIndexCorrectnessMetric
from unitxt.test_utils.metrics import test_metric

# metric = MetricPipeline(
#     main_score="LlamaIndexCorrectness",
#     # preprocess_steps=[
#     #     CopyFields(field_to_field=[("references/0", "references")], use_query=True),
#     #     CastFields(
#     #         fields={"prediction": "float", "references": "float"},
#     #         failure_defaults={"prediction": None},
#     #         use_nested_query=True,
#     #     ),
#     # ],
#     metric=LlamaIndexCorrectnessMetric(),
#     postpreprocess_steps=[
#         # CopyFields(
#         #     field_to_field=[
#         #         ("score/instance/score", "score"),
#         #     ],
#         #     use_query=True,
#         # )
#     ],
# )

metric = LlamaIndexCorrectnessMetric()

predictions = ["The right answer"]
references = [["The right answer"]]
inputs = [
    {
        "question": "question number 1",
        "contexts": ["context number 1"],
        "reference_answer": "The right answer",
    }
]

instance_targets = [  # nDCG is undefined at instance level
    {
        "score": 1.0,
        "score_name": "score",
        "feedback": "The generated answer is fully correct and relevant to the user query, matching the reference answer exactly.",
    }
] * len(predictions)

global_target = {"score": 1.0, "score_name": "score"}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    additional_inputs=inputs,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.rag.llama_index_correctness", overwrite=True)
