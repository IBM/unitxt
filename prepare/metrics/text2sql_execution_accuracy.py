from unitxt.catalog import add_to_catalog
from unitxt.metrics import ExecutionAccuracy
from unitxt.test_utils.metrics import test_metric

metric = ExecutionAccuracy()

predictions = [
    "SELECT nme FROM employees WHERE department = 'Sales'",
    "SELECT name FROM employees WHERE department = 'Sales'",
]  # Incorrect column name 'nme'
references = [["SELECT name FROM employees WHERE department = 'Sales';"]] * 2
task_data = [
    {
        "db": {
            "db_id": "mock_db",
            "db_type": "in_memory",
            "data": {
                "employees": {
                    "columns": ["id", "name", "department", "salary"],
                    "rows": [
                        (1, "Alice", "Sales", 50000),
                        (2, "Bob", "Engineering", 60000),
                        (3, "Charlie", "Sales", 55000),
                    ],
                }
            },
        }
    }
] * 2

instance_targets = [
    {
        "execution_accuracy": 0.0,
        "num_of_instances": 1,
        "score": 0.0,
        "score_name": "execution_accuracy",
    },
    {
        "execution_accuracy": 1.0,
        "num_of_instances": 1,
        "score": 1.0,
        "score_name": "execution_accuracy",
    },
]


global_target = {
    "execution_accuracy": 0.5,
    "execution_accuracy_ci_high": 1.0,
    "execution_accuracy_ci_low": 0.0,
    "num_of_instances": 2,
    "score": 0.5,
    "score_ci_high": 1.0,
    "score_ci_low": 0.0,
    "score_name": "execution_accuracy",
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
    task_data=task_data,
)

add_to_catalog(metric, "metrics.text2sql.execution_accuracy", overwrite=True)
