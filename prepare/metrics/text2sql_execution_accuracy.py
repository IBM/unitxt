from unitxt.catalog import add_to_catalog
from unitxt.test_utils.metrics import test_metric
from unitxt.text2sql.metrics import ExecutionAccuracy

metric = ExecutionAccuracy()

predictions = [
    "SELECT ROUND(MAX((`Free Meal Count (K-12)` / `Enrollment (K-12)`) * 100), 2) AS Highest_Free_Rate\nFROM frpm\nWHERE `County Name` = 'Alameda';",
    "SELECT `Educational Option Type`, \n        ROUND((`Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`), 4) AS Eligible_Free_Rate\nFROM    frpm\nWHERE   `Educational Option Type` = 'Continuation'\nORDER BY Eligible_Free_Rate\nLIMIT 3;\n\nExplanation:\nThis SQL query first filters the data to only include continuation schools. Then, it calculates the eligible free rate for each school by dividing the free meal count by the enrollment count. The result is rounded to four decimal places. Finally, the query orders the results by the eligible free rate in ascending order and limits the output to the lowest three rates.",
    "SELECT s.Zip\nFROM frpm f\nJOIN schools s ON f.CDSCode = s.CDSCode\nWHERE f.`Charter School (Y/N)` = 1 AND f.`District Name` LIKE '%Fresno County Office of Education%';\n\nExplanation:\nThis SQL query joins the frpm and schools tables on the CDSCode column. It then filters the results to only include charter schools (where `Charter School (Y/N)` = 1) and schools that are part of the Fresno County Office of Education district (where `District Name` contains the specified string). Finally, it selects the Zip column from the schools table, which contains the zip code of each charter school in Fresno County Office of Education.",
    "SELECT s.MailStreet\nFROM    frpm f\nJOIN    schools s ON f.CDSCode = s.CDSCode\nORDER BY f.`FRPM Count (K-12)` DESC\nLIMIT 1;\n```\n\nThis SQL query will return the unabbreviated mailing address of the school with the highest FRPM count for K-12 students. Here's how it works:\n\n1. The query joins the `frpm` table with the `schools` table on the `CDSCode` column, which is common to both tables.\n2. The query orders the joined table by the `FRPM Count (K-12)` column in descending order (highest count first).\n3. The `LIMIT 1` clause returns only the top row, which corresponds to the school with the highest FRPM count.\n4. The `SELECT` clause returns the `MailStreet` column, which contains the unabbreviated mailing address of the school.\n\nNote that if there are multiple schools with the same highest FRPM count, this query will return only one of them. If you want to return all schools with the highest FRPM count, you can modify the query to use a subquery or a window function.",
]
references = [
    [
        "SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1"
    ],
    [
        "SELECT `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` FROM frpm WHERE `Educational Option Type` = 'Continuation School' AND `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` IS NOT NULL ORDER BY `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` ASC LIMIT 3"
    ],
    [
        "SELECT T2.Zip FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`District Name` = 'Fresno County Office of Education' AND T1.`Charter School (Y/N)` = 1"
    ],
    [
        "SELECT T2.MailStreet FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode ORDER BY T1.`FRPM Count (K-12)` DESC LIMIT 1"
    ],
]
task_data = [
    {
        "_id": "0",
        "question": "What is the highest eligible free rate for K-12 students in the schools in Alameda County?",
        "db_id": "california_schools",
    },
    {
        "_id": "1",
        "question": "Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.",
        "db_id": "california_schools",
    },
    {
        "_id": "2",
        "question": "Please list the zip code of all the charter schools in Fresno County Office of Education.",
        "db_id": "california_schools",
    },
    {
        "_id": "3",
        "question": "What is the unabbreviated mailing address of the school with the highest FRPM count for K-12 students?",
        "db_id": "california_schools",
    },
]

instance_targets = [
    {
        "execution_accuracy": 0.0,
        "score": 0.0,
        "score_name": "execution_accuracy",
    },
    {
        "execution_accuracy": 0.0,
        "score": 0.0,
        "score_name": "execution_accuracy",
    },
    {
        "execution_accuracy": 1.0,
        "score": 1.0,
        "score_name": "execution_accuracy",
    },
    {
        "execution_accuracy": 1.0,
        "score": 1.0,
        "score_name": "execution_accuracy",
    },
]

global_target = {
    "execution_accuracy": 0.5,
    "execution_accuracy_ci_low": 0.0,
    "execution_accuracy_ci_high": 1.0,
    "score": 0.5,
    "score_ci_low": 0.0,
    "score_ci_high": 1.0,
    "score_name": "execution_accuracy",
    "num_of_instances": 4,
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
