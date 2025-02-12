from unitxt.catalog import add_to_catalog
from unitxt.metrics import SQLExecutionAccuracy
from unitxt.test_utils.metrics import test_metric

metric = SQLExecutionAccuracy()

predictions = [
    "SELECT nme FROM employees WHERE department = 'Sales'",
    "SELECT name FROM employees WHERE department = 'Sales'",
    "SELECT name FROM employees WHERE department = 'Engineering'",
    "SELECT id, name FROM employees WHERE department = 'Sales'",
    "SELECT name FROM employees WHERE department = 'Non-Existent'",
]  # Incorrect column name 'nme'
references = [
    ["SELECT name FROM employees WHERE department = 'Sales';"],
    ["SELECT name FROM employees WHERE department = 'Sales';"],
    ["SELECT name FROM employees WHERE department = 'Sales';"],
    ["SELECT name FROM employees WHERE department = 'Sales';"],
    ["SELECT name FROM employees WHERE department = 'Non-Existent';"],
]
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
] * 5

instance_targets = [
    {
        "error_message": "Error executing SQL: no such column: nme",
        "execution_accuracy": 0.0,
        "gold_df_json": "",
        "gold_error": 0.0,
        "non_empty_execution_accuracy": 0.0,
        "non_empty_gold_df": 0.0,
        "predicted_df_json": "",
        "predicted_error": 1.0,
        "score": 0.0,
        "score_name": "non_empty_execution_accuracy",
        "subset_non_empty_execution_result": 0.0,
    },
    {
        "error_message": "",
        "execution_accuracy": 1.0,
        "gold_df_json": '{"0":{"0":"Alice","1":"Charlie"}}',
        "gold_error": 1.0,
        "non_empty_execution_accuracy": 1.0,
        "non_empty_gold_df": 1.0,
        "predicted_df_json": '{"0":{"0":"Alice","1":"Charlie"}}',
        "predicted_error": 0.0,
        "score": 1.0,
        "score_name": "non_empty_execution_accuracy",
        "subset_non_empty_execution_result": 1.0,
    },
    {
        "error_message": "None",
        "execution_accuracy": 0.0,
        "gold_df_json": '{"0":{"0":"Alice","1":"Charlie"}}',
        "gold_error": 0.0,
        "non_empty_execution_accuracy": 0.0,
        "non_empty_gold_df": 1.0,
        "predicted_df_json": '{"0":{"0":"Bob"}}',
        "predicted_error": 0.0,
        "score": 0.0,
        "score_name": "non_empty_execution_accuracy",
        "subset_non_empty_execution_result": 0.0,
    },
    {
        "error_message": "None",
        "execution_accuracy": 0.0,
        "gold_df_json": '{"0":{"0":"Alice","1":"Charlie"}}',
        "gold_error": 0.0,
        "non_empty_execution_accuracy": 0.0,
        "non_empty_gold_df": 1.0,
        "predicted_df_json": '{"0":{"0":1,"1":3},"1":{"0":"Alice","1":"Charlie"}}',
        "predicted_error": 0.0,
        "score": 0.0,
        "score_name": "non_empty_execution_accuracy",
        "subset_non_empty_execution_result": 1.0,
    },
    {
        "error_message": "",
        "execution_accuracy": 1.0,
        "gold_df_json": "{}",
        "gold_error": 1.0,
        "non_empty_execution_accuracy": 0.0,
        "non_empty_gold_df": 0.0,
        "predicted_df_json": "{}",
        "predicted_error": 0.0,
        "score": 0.0,
        "score_name": "non_empty_execution_accuracy",
        "subset_non_empty_execution_result": 0.0,
    },
]


global_target = {
    "execution_accuracy": 0.4,
    "execution_accuracy_ci_high": 0.87,
    "execution_accuracy_ci_low": 0.0,
    "gold_error": 0.4,
    "gold_sql_runtime_ci_high": 0.0,
    "gold_sql_runtime_ci_low": 0.0,
    "non_empty_execution_accuracy": 0.2,
    "non_empty_execution_accuracy_ci_high": 0.8,
    "non_empty_execution_accuracy_ci_low": 0.0,
    "non_empty_gold_df": 0.6,
    "num_of_instances": 5,
    "predicted_error": 0.2,
    "predicted_sql_runtime_ci_high": 0.0,
    "predicted_sql_runtime_ci_low": 0.0,
    "score": 0.2,
    "score_ci_high": 0.8,
    "score_ci_low": 0.0,
    "score_name": "non_empty_execution_accuracy",
    "subset_non_empty_execution_result": 0.4,
    "subset_non_empty_execution_result_ci_high": 1.0,
    "subset_non_empty_execution_result_ci_low": 0.0,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
    task_data=task_data,
    score_keys_to_ignore=[
        "predicted_sql_runtime",
        "gold_sql_runtime",
        "pred_to_gold_runtime_ratio",
    ],
)

add_to_catalog(metric, "metrics.text2sql.execution_accuracy", overwrite=True)
