from unitxt.catalog import add_to_catalog
from unitxt.metrics import SQLExecutionAccuracy, SQLNonExecutionAccuracy
from unitxt.test_utils.metrics import test_metric

sql_execution_accuracy_metric = SQLExecutionAccuracy()

predictions = [
    "SELECT nme FROM employees WHERE department = 'Sales'",
    "SELECT name FROM employees WHERE department = 'Sales'",
    "SELECT name FROM employees WHERE department = 'Engineering'",
    "SELECT id, name FROM employees WHERE department = 'Sales'",
    "SELECT name FROM employees WHERE department = 'Non-Existent'",
    "Garbage SELECT *",
]  # Incorrect column name 'nme'
references = [
    ["SELECT name FROM employees WHERE department = 'Sales';"],
    ["SELECT name FROM employees WHERE department = 'Sales';"],
    ["SELECT name FROM employees WHERE department = 'Sales';"],
    ["SELECT name FROM employees WHERE department = 'Sales';"],
    ["SELECT name FROM employees WHERE department = 'Non-Existent';"],
    ["SELECT name FROM employees WHERE department = 'Sales';"],
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
] * 6

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
    {
        "error_message": "Error executing SQL: no tables specified",
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
]


global_target = {
    "execution_accuracy": 0.33,
    "execution_accuracy_ci_high": 0.83,
    "execution_accuracy_ci_low": 0.0,
    "gold_error": 0.33,
    "gold_sql_runtime_ci_high": 0.0,
    "gold_sql_runtime_ci_low": 0.0,
    "non_empty_execution_accuracy": 0.17,
    "non_empty_execution_accuracy_ci_high": 0.67,
    "non_empty_execution_accuracy_ci_low": 0.0,
    "non_empty_gold_df": 0.5,
    "num_of_instances": 6,
    "predicted_error": 0.33,
    "predicted_sql_runtime_ci_high": 0.0,
    "predicted_sql_runtime_ci_low": 0.0,
    "score": 0.17,
    "score_ci_high": 0.67,
    "score_ci_low": 0.0,
    "score_name": "non_empty_execution_accuracy",
    "subset_non_empty_execution_result": 0.33,
    "subset_non_empty_execution_result_ci_high": 0.83,
    "subset_non_empty_execution_result_ci_low": 0.0,
}

outputs = test_metric(
    metric=sql_execution_accuracy_metric,
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

add_to_catalog(
    sql_execution_accuracy_metric, "metrics.text2sql.execution_accuracy", overwrite=True
)

sql_non_execution_accuracy_metric = SQLNonExecutionAccuracy()

instance_targets = [
    {
        "score": 0.0,
        "score_name": "sqlglot_equivalence",
        "sql_exact_match": 0.0,
        "sqlglot_equivalence": 0.0,
        "sqlglot_optimized_equivalence": 0.0,
        "sqlglot_validity": 1.0,
        "sqlparse_equivalence": 0.0,
        "sqlparse_validity": 1.0,
    },
    {
        "score": 1.0,
        "score_name": "sqlglot_equivalence",
        "sql_exact_match": 1.0,
        "sqlglot_equivalence": 1.0,
        "sqlglot_optimized_equivalence": 1.0,
        "sqlglot_validity": 1.0,
        "sqlparse_equivalence": 0.0,
        "sqlparse_validity": 1.0,
    },
    {
        "score": 0.0,
        "score_name": "sqlglot_equivalence",
        "sql_exact_match": 0.0,
        "sqlglot_equivalence": 0.0,
        "sqlglot_optimized_equivalence": 0.0,
        "sqlglot_validity": 1.0,
        "sqlparse_equivalence": 0.0,
        "sqlparse_validity": 1.0,
    },
    {
        "score": 0.0,
        "score_name": "sqlglot_equivalence",
        "sql_exact_match": 0.0,
        "sqlglot_equivalence": 0.0,
        "sqlglot_optimized_equivalence": 0.0,
        "sqlglot_validity": 1.0,
        "sqlparse_equivalence": 0.0,
        "sqlparse_validity": 1.0,
    },
    {
        "score": 1.0,
        "score_name": "sqlglot_equivalence",
        "sql_exact_match": 1.0,
        "sqlglot_equivalence": 1.0,
        "sqlglot_optimized_equivalence": 1.0,
        "sqlglot_validity": 1.0,
        "sqlparse_equivalence": 0.0,
        "sqlparse_validity": 1.0,
    },
    {
        "score": 0.0,
        "score_name": "sqlglot_equivalence",
        "sql_exact_match": 0.0,
        "sqlglot_equivalence": 0.0,
        "sqlglot_optimized_equivalence": 0.0,
        "sqlglot_validity": 1.0,
        "sqlparse_equivalence": 0.0,
        "sqlparse_validity": 1.0,
    },
]


global_target = {
    "num_of_instances": 6,
    "score": 0.33,
    "score_ci_high": 0.83,
    "score_ci_low": 0.0,
    "score_name": "sqlglot_equivalence",
    "sql_exact_match": 0.33,
    "sql_exact_match_ci_high": 0.83,
    "sql_exact_match_ci_low": 0.0,
    "sqlglot_equivalence": 0.33,
    "sqlglot_equivalence_ci_high": 0.83,
    "sqlglot_equivalence_ci_low": 0.0,
    "sqlglot_optimized_equivalence": 0.33,
    "sqlglot_optimized_equivalence_ci_high": 0.83,
    "sqlglot_optimized_equivalence_ci_low": 0.0,
    "sqlglot_validity": 1.0,
    "sqlparse_equivalence": 0.0,
    "sqlparse_validity": 1.0,
}

outputs = test_metric(
    metric=sql_non_execution_accuracy_metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
    task_data=task_data,
)

add_to_catalog(
    sql_non_execution_accuracy_metric,
    "metrics.text2sql.non_execution_accuracy",
    overwrite=True,
)
