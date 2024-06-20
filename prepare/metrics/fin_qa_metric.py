from typing import Dict, List

import numpy as np
from fin_qa_eval_script import equal_program, eval_program, program_tokenization
from unitxt import add_to_catalog
from unitxt.metrics import InstanceMetric


class FinQAEval(InstanceMetric):
    reduction_map = {"mean": ["accuracy"]}
    main_score = "accuracy"
    ci_scores = ["accuracy"]

    def finqa_eval(
        self, references: List[List], prediction: str, task_data: Dict
    ) -> float:
        exe_correct = False
        prog_correct = False

        pred_item = program_tokenization(prediction)
        gold_answer = references[0]
        table = task_data["table"]
        for program in task_data["program_re"]:
            gold = program_tokenization(program)
            invalid_flag, exe_res = eval_program(pred_item, table)
            if invalid_flag == 0 and exe_res == gold_answer:
                exe_correct = True
            if equal_program(gold, pred_item) and exe_res == gold_answer:
                prog_correct = True

        return exe_correct and prog_correct

    def python_expression_eval(
        self, references: List[List], prediction: str, task_data: Dict
    ) -> float:
        total = 0
        correct = 0

        for pred, gold_item in zip(prediction, references):
            if pred.lower().endswith(gold_item.lower()):
                # for non numeric answers, just check if the answer is in the prediction
                correct += 1
            else:
                # first remove all percent signs and money signs from the answer
                pred = pred.replace("%", "").replace("$", "")
                # if it contains an equal sign, take the part before the equal sign
                if "=" in pred:
                    pred = pred.split("=")[0]

                # if gold is a percentage, remove the percent sign and express as a decimal
                if gold_item.endswith("%"):
                    gold = float(gold_item.replace("%", "")) / 100
                # try to evaluate the expression
                else:
                    try:
                        # not a percentage, and can't be converted to a float
                        gold = float(eval(gold_item))
                    except:
                        pass
                try:
                    pred = float(eval(pred))
                    # round to the same number of decimal places as the gold answer
                    pred = round(pred, len(str(gold).split(".")[1]))
                    # if the prediction is close enough to the gold answer, count as correct
                    if np.isclose(pred, gold, atol=0.001):
                        correct += 1
                except:
                    # count as incorrect
                    pass
            total += 1
        return float(correct) / total

    def compute(self, references: List[List], prediction: str, task_data: Dict) -> dict:
        acc = 0
        try:
            acc = self.finqa_eval(references, prediction, task_data)
        except:
            # fall back to evaluating the python expression.
            acc = max(
                self.python_expression_eval(references, prediction, task_data), acc
            )
        return {self.main_score: acc}


metric = FinQAEval()
add_to_catalog(metric, "metrics.fin_qa_metric", overwrite=True)
