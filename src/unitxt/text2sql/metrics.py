import sqlite3
from typing import Any, Dict, List, Tuple

import evaluate
from func_timeout import FunctionTimedOut, func_timeout

from ..metrics import InstanceMetric
from .data_utils import SQLData

SQL_TIMEOUT = 350.0

logger = evaluate.logging.get_logger(__name__)


## function below from BIRD repo https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/evaluation.py
def run_sql_and_match(predicted_sql: str, ground_truth: str, db_path: str) -> int:
    res: int = 0
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    logger.debug(f"Running SQL query over SQLite DB: {db_path}")
    try:
        cursor.execute(predicted_sql)
        predicted_res: List[Tuple] = cursor.fetchall()
    except sqlite3.Error:
        return 0
    cursor.execute(ground_truth)
    ground_truth_res: List[Tuple] = cursor.fetchall()

    logger.debug(f"predicted_res: {predicted_res}")
    logger.debug(f"ground_truth_sql: {ground_truth}")
    logger.debug(f"ground_truth_res: {ground_truth_res}")

    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res


class ExecutionAccuracy(InstanceMetric):
    reduction_map = {"mean": ["execution_accuracy"]}
    main_score = "execution_accuracy"
    ci_scores = ["execution_accuracy"]

    prediction_type = "Any"  # string representation is compared
    sql_data = SQLData()
    metric_flavour = "bird"

    def compute(self, references: List[Any], prediction: str, task_data: Dict) -> dict:
        predicted_sql = prediction
        execution_result: float = 0.0

        if predicted_sql and predicted_sql.strip() != "":
            if not predicted_sql.startswith("SELECT") and "SELECT" in predicted_sql:
                predicted_sql = predicted_sql[predicted_sql.find("SELECT") :]
            if ";" in predicted_sql:
                predicted_sql = predicted_sql[: predicted_sql.find(";") + 1]
            db_id = task_data["db_id"]
            db_file_path = self.sql_data.get_db_file_path(db_id)
            logger.debug(f"Database file: {db_file_path}")
            try:
                execution_result = func_timeout(
                    SQL_TIMEOUT,
                    run_sql_and_match,
                    args=(
                        predicted_sql,
                        references[0],
                        db_file_path,
                    ),
                )
            except FunctionTimedOut:
                logger.error("QUERY TIMEOUT")
                pass

        result = {self.main_score: float(execution_result)}
        logger.debug(f"Result: {result}")
        result["score"] = result[self.main_score]
        result["score_name"] = self.main_score
        return result
