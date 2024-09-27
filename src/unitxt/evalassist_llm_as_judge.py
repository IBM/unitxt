from .metrics import BulkInstanceMetric
from .inference import InferenceEngine
from .templates import Template
from .task import Task
from .artifact import Artifact
from .api import infer

from typing import List,Dict, Literal

class RubricOption:
    def __init__(self, option: str, description: str):
        self.option = option
        self.description = description

class Rubric(Artifact):
    name: str
    criteria: str
    options: List[Dict[Literal["option", "description"], str]]

class EvalAssistLLMAsJudge(BulkInstanceMetric):
    inference_model: InferenceEngine
    rubric: Rubric = None
    assessment_template : Template = None
    summ_template : Template = None
    answer_template : Template = None

    reduction_map = {"mean": ["score"]}
    main_score = "score"

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

        if self.inference_model.model_name == "kaist-ai/prometheus-8x7b-v2":
            # NOTE : EvalAssist has rubric_options : list(list(RubricOption))
            # score_options = [[o.option for o in criteria_option_list] for criteria_option_list in rubric_options]
            # score_rubric = ["".join([f"Score {o.option}: {o.description}\n" for o in criteria_option_list]) for criteria_option_list in rubric_options]
            score_options = [o.option for o in rubric_options]
            score_rubric = ["".join([f"Score {o.option}: {o.description}\n" for o in rubric_options])]
            return criteria, score_options, score_rubric
        return options_as_str, criteria
        
    def prepare(self):
        # TODO: if we have predefined rubrics --> prepare should create it only once
        # if self.rubric:
        #     self.set_rubric()
            
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

        if self.inference_model.model_name == "kaist-ai/prometheus-8x7b-v2":
            criteria, score_options, score_rubric = self.set_rubric(rubric)

            # Assessment Stage
            instances = [{
                       "score_instructions" : " or ".join(score_options_list),
                        "context_variables": input_instance['question'],
                        "response": prediction,
                        "criteria": criterion,
                        "score_rubric" : "".join(score_rubric*len(predictions))
                    }
                    for input_instance, prediction, criterion, score_options_list in zip(
                        task_data, predictions, [criteria] * len(predictions), [score_options] * len(predictions))]
            
            assessment_outputs = infer(instances, task=self.assessment_task_prometheus, engine=self.inference_model, template=self.assessment_template)
            print("ass_output prometheus", assessment_outputs)

            feedbacks = [out.split("[RESULT]")[0].strip() for out in assessment_outputs]
            return [{"score": 0.8, "assessment": assessment_outputs[i], "feedbacks": feedbacks[i]} for i in range(len(predictions))]
        else:
            options_as_str, criteria = self.set_rubric(rubric)
            # Assessment Stage
            instances = [{
                        "context_variables": input_instance['question'],
                        "response": prediction,
                        "criteria": criterion,
                        "options": option
                    }
                    for input_instance, prediction, criterion, option in zip(
                        task_data, predictions, [criteria] * len(predictions), [options_as_str] * len(predictions))]
            
            assessment_outputs = infer(instances, task=self.assessment_task, engine=self.inference_model, template=self.assessment_template)
            print("ass_output ", assessment_outputs)
            
            # Summarisation Stage
            assessment_instances = [{"assessment": assessment_output} for assessment_output in assessment_outputs]
            summ_output = infer(assessment_instances, task=self.summ_task, engine=self.inference_model, template=self.summ_template)
            
            print("summ_output ", summ_output)

            return [{"score": 0.8, "assessment": assessment_outputs[i], "summary": summ_output[i]} for i in range(len(predictions))]
    