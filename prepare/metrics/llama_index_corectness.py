import numpy as np
from unitxt.metrics import LlamaIndexCorrectnessMetric, MetricPipeline
from unitxt.test_utils.metrics import test_metric

metric = MetricPipeline(
    main_score="LlamaIndexCorrectness",
    # preprocess_steps=[
    #     CopyFields(field_to_field=[("references/0", "references")], use_query=True),
    #     CastFields(
    #         fields={"prediction": "float", "references": "float"},
    #         failure_defaults={"prediction": None},
    #         use_nested_query=True,
    #     ),
    # ],
    metric=LlamaIndexCorrectnessMetric(),
    postpreprocess_steps=[
        # CopyFields(
        #     field_to_field=[
        #         ("score/instance/score", "score"),
        #     ],
        #     use_query=True,
        # )
    ],
)


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
    {"nDCG": np.nan, "score": np.nan, "score_name": "nDCG"}
] * len(predictions)

global_target = {
    "nDCG": 0.42,
    "nDCG_ci_high": 0.66,
    "nDCG_ci_low": 0.15,
    "score": 0.42,
    "score_ci_high": 0.66,
    "score_ci_low": 0.15,
    "score_name": "nDCG",
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    additional_inputs=inputs,
    instance_targets=instance_targets,
    global_target=global_target,
)
