from unitxt import add_to_catalog
from unitxt.metrics import FaithfulnessHHEM, MetricPipeline
from unitxt.operators import Copy
from unitxt.test_utils.metrics import test_metric

pairs = [
    ("The capital of France is Berlin.", "The capital of France is Paris."),
    ("I am in California", "I am in United States."),
    ("I am in United States", "I am in California."),
]

task_data = [{"contexts": [p[0]], "answer": p[1]} for p in pairs]

metric = MetricPipeline(
    main_score="score",
    preprocess_steps=[
        Copy(
            field_to_field={
                "task_data/contexts": "contexts",
                "task_data/answer": "answer",
            },
            not_exist_do_nothing=True,
        ),
        Copy(field_to_field={"contexts": "references", "answer": "prediction"}),
    ],
    metric=FaithfulnessHHEM(),
)
instance_targets = [
    {"score": 0.01, "score_name": "score"},
    {"score": 0.65, "score_name": "score"},
    {"score": 0.13, "score_name": "score"},
]
global_target = {
    "num_of_instances": 3,
    "score": 0.26,
    "score_name": "score",
    "score_ci_low": 0.05,
    "score_ci_high": 0.65,
}
outputs = test_metric(
    metric=metric,
    predictions=[""] * len(instance_targets),
    references=[[""]] * len(instance_targets),
    task_data=task_data,
    instance_targets=instance_targets,
    global_target=global_target,
)
add_to_catalog(metric, "metrics.rag.faithfulness.vectara_hhem_2_1", overwrite=True)
