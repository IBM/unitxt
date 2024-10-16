from .metrics import BulkInstanceMetric
from .inference import InferenceEngine
from .templates import Template
from .task import Task
from .artifact import Artifact
from .api import infer, infer_log_probs
from unitxt import get_logger

import re
import random
from typing import List,Dict, Literal
import numpy as np

class RubricOption:
    def __init__(self, option: str, description: str):
        self.option = option
        self.description = description

class Rubric(Artifact):
    name: str
    criteria: str
    options: List[Dict[Literal["option", "description"], str]]

class EvalAssistLLMAsJudgeDirect(BulkInstanceMetric):
    inference_model: InferenceEngine
    rubric: Rubric = None
    assessment_template : Template = None
    summ_template : Template = None
    answer_template : Template = None

    reduction_map = {"mean": ["score"]}
    main_score = "score"
    logger = get_logger()

    assessment_task =  Task(
            input_fields={"context_variables": str, "response": str, "criteria": str, "options": str},
            reference_fields={},
            prediction_type=str,
            metrics=[])

    assessment_task_prometheus =  Task(
            input_fields={"score_instructions" : str, "context_variables": str,
                        "response": str, "criteria": str, "score_rubric" : str},
            reference_fields={},
            prediction_type=str,
            metrics=[])

    summ_task =  Task(
            input_fields={"assessment": str},
            reference_fields={},
            prediction_type=str,
            metrics=[])

    def set_rubric(self, rubrics):
        rubric_options = [RubricOption(rub_option["option"], rub_option["description"]) for rub_option in rubrics.options]
        options_as_str = "Choose an answer:\n" + "\n".join([f"- \"{o.option}\"{f' if {o.description}' if o.description != '' else ''}" for o in rubric_options])
        criteria = rubrics.criteria

        if self.inference_model.model_name == "kaist-ai/prometheus-8x7b-v2" or self.inference_model.model_name == "gpt-4o-2024-05-13":
            # NOTE : EvalAssist has rubric_options : list(list(RubricOption))
            # score_options = [[o.option for o in criteria_option_list] for criteria_option_list in rubric_options]
            # score_rubric = ["".join([f"Score {o.option}: {o.description}\n" for o in criteria_option_list]) for criteria_option_list in rubric_options]
            score_options = [o.option for o in rubric_options]
            score_rubric = "".join([f"Score {o.option}: {o.description}\n" for o in rubric_options])
            return criteria, score_options, score_rubric
        return options_as_str, criteria

    def _parse_completion(self, completion: str, options: list[str]) -> tuple[str, str]:
        """ Ensure that the assessments are always a valid option """
        for o in options:
            search_for = rf"\b{o.strip()}\b"
            match_found = re.search(search_for, completion)
            if match_found is not None:
                return match_found[0]
        # failure case - return a arbitrary option label
        return random.choice(options)

        
    def prepare(self):
        super().prepare()

    def compute(
        self, references: List[List[str]], predictions: List[str], task_data: List[Dict[str,str]]
    ) -> dict:
        if self.rubric is None:
           # Get it from the task data
           rubric_dict = task_data[0]["rubric"]
           rubric = Rubric(**rubric_dict)
        else:
            rubric = self.rubric
        
        # gpt-4o-2024-05-13
        if self.inference_model.model_name == "gpt-4o-2024-05-13":
            criteria, score_options, options_as_str = self.set_rubric(rubric)

            instances = [{
                        "context_variables": input_instance['context'],
                        "response": prediction,
                        "criteria": criterion,
                        "options": option
                    }
                    for input_instance, prediction, criterion, option in zip(
                        task_data, predictions, [criteria] * len(predictions), [options_as_str] * len(predictions))]

            generated_text_and_logprobs = infer_log_probs(instances, task=self.assessment_task, engine=self.inference_model, template=self.assessment_template)
            generated_texts = [completion['text'] for completion in generated_text_and_logprobs]

            # Assessment stage
            assessments = [self._parse_completion(text_completion, score_option) for text_completion, score_option in zip(generated_texts, score_options* len(predictions))]
            certainties = [float(np.mean([np.exp(logprob.logprob) for logprob in completion['logprobs']])) for i, completion in enumerate(generated_text_and_logprobs)]
            self.logger.info("generated assessment")
 
            # Summarization stage
            assessment_instances = [{"assessment": assessment_output} for assessment_output in generated_texts]
            summ_output = infer(assessment_instances, task=self.summ_task, engine=self.inference_model, template=self.summ_template)
            self.logger.info("generated summary")

            return [{"score": 0.8, "assessment": assessments[0], "summary": summ_output[i], "certainties": certainties[0] } for i in range(len(predictions))]

        elif self.inference_model.model_name == "kaist-ai/prometheus-8x7b-v2":
            criteria, score_options, score_rubric = self.set_rubric(rubric)

            # Assessment Stage
            instances = [{
                       "score_instructions" : " or ".join(score_options_list),
                        "context_variables": input_instance['context'],
                        "response": prediction,
                        "criteria": criterion,
                        "score_rubric" : "".join([score_rubric]*len(predictions))
                    }
                    for input_instance, prediction, criterion, score_options_list in zip(
                        task_data, predictions, [criteria] * len(predictions), [score_options] * len(predictions))]

            assessment_outputs = infer(instances, task=self.assessment_task_prometheus, engine=self.inference_model, template=self.assessment_template)
            self.logger.info("generated assessment")

            feedbacks = [out.split("[RESULT]")[0].strip() for out in assessment_outputs]
            self.logger.info("generated feedback")
            return [{"score": 0.8, "assessment": assessment_outputs[i], "feedbacks": feedbacks[i]} for i in range(len(predictions))]
        else:
            options_as_str, criteria = self.set_rubric(rubric)

            # Assessment Stage
            instances = [{
                        "context_variables": input_instance['context'],
                        "response": prediction,
                        "criteria": criterion,
                        "options": option
                    }
                    for input_instance, prediction, criterion, option in zip(
                        task_data, predictions, [criteria] * len(predictions), [options_as_str] * len(predictions))]

            assessment_outputs = infer(instances, task=self.assessment_task, engine=self.inference_model, template=self.assessment_template)
            self.logger.info("generated assessment")

            # Summarisation Stage
            assessment_instances = [{"assessment": assessment_output} for assessment_output in assessment_outputs]
            summ_output = infer(assessment_instances, task=self.summ_task, engine=self.inference_model, template=self.summ_template)
            self.logger.info("generated summary")

            return [{"score": 0.8, "assessment": assessment_outputs[i], "summary": summ_output[i]} for i in range(len(predictions))]
        