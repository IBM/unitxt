import re
import random
import numpy as np
from typing import List,Dict
from .metrics import BulkInstanceMetric
from .inference import InferenceEngine
from unitxt import get_logger
from .templates import Template
from .task import Task
from .artifact import Artifact
from .api import infer, infer_log_probs
from .formats import OpenAIFormat

class PairwiseCriteria(Artifact):
    name : str
    criteria : str

class EvalAssistLLMAsJudgePairwise(BulkInstanceMetric):
    inference_model: InferenceEngine
    pairwise_criteria: PairwiseCriteria = None
    assessment_template : Template = None
    summ_template : Template = None
    answer_template : Template = None

    reduction_map = {"mean": ["score"]}
    main_score = "score"
    logger = get_logger()

    assessment_task =  Task(
            input_fields={"context_variables": str, "response_a" : str, "response_b" : str,
                "option_a" : str, "option_b" : str, "criteria_name" : str, "criteria_description" : str},
            reference_fields={},
            prediction_type=str,
            metrics=[])
    
    assessment_task_prometheus =  Task(
            input_fields={"context_variables": str, "response_a" : str, "response_b" : str,
                "option_a" : str, "option_b" : str, "rubric": str},
            reference_fields={},
            prediction_type=str,
            metrics=[])
    
    summ_task =  Task(
            input_fields={"assessment": str},
            reference_fields={},
            prediction_type=str,
            metrics=[])
    
    def _parse_completion(self, completion: str, option_pair: list[str]) -> tuple[str, str]:
        """ Ensure that the assessments are always a valid option """
        quotes_pattern = r'(""".*?"""|```.*?```)'
        winner_pattern = r'Winner:\s*(.*)'
        # try to find the triple quotes section
        match = re.findall(quotes_pattern, completion)
        # if it doesnt exist, match for winner
        if len(match) == 0:
            match = re.findall(winner_pattern, completion)
        if len(match) > 0:
            decision = match[0]
            for o in option_pair:
                search_for = rf"\b{o.strip()}\b"
                match_found = re.search(search_for, decision)
                if match_found is not None:
                    return match_found[0]
        
        # failure case - return a arbitrary option label
        return random.choice(option_pair)
   
    def prepare(self):
        super().prepare()

    def compute(
        self, references: List[List[str]], predictions: List[str], task_data: List[Dict[str,str]]
    ) -> dict:
        
        if self.pairwise_criteria is None:
            #Get it from the task data
            pairwise_criteria_dict = task_data[0]["pairwise_criteria"]
            pairwise_criteria = PairwiseCriteria(**pairwise_criteria_dict)
        else:
            pairwise_criteria = self.pairwise_criteria

        #TODO: Not sure, have to check again
        option_pairs = [['1', '2'] for _ in range(len(predictions))]
        kwargs = {}

        if self.inference_model.model_name == "gpt-4o-2024-05-13":
            kwargs["format"] = OpenAIFormat()
            assessment_prompt_user_content_instances = [{
                    "context_variables": input_instance['context'],
                    "response_a" : prediction[0],
                    "response_b" : prediction[1],
                    "option_a" : option_pair[0],
                    "option_b" : option_pair[1],
                    "criteria_name" : pairwise_criteria.name,
                    "criteria_description" : pairwise_criteria.criteria
                } for prediction, option_pair, input_instance in zip(predictions, option_pairs, task_data)]
            
            generated_text_and_logprobs = infer_log_probs(assessment_prompt_user_content_instances, task=self.assessment_task, engine=self.inference_model, template=self.assessment_template, **kwargs)
            generated_texts = [completion['text'] for completion in generated_text_and_logprobs]

            # Assessment stage
            assessments = [self._parse_completion(text_completion, option_pair) for text_completion, option_pair in zip(generated_texts, option_pairs*len(predictions))]

            certainties = [float(np.mean([np.exp(logprob.logprob) for logprob in completion['logprobs']])) for i, completion in enumerate(generated_text_and_logprobs)]
            self.logger.info("generated assessment")
 
            # Summarization stage
            assessment_instances = [{"assessment": assessment_output} for assessment_output in generated_texts]
            summ_output = infer(assessment_instances, task=self.summ_task, engine=self.inference_model, template=self.summ_template)
            self.logger.info("generated summary")

            return [{"score": 0.8, "assessment": assessments[i], "summary": summ_output[i], "certainties": certainties[i] } for i in range(len(predictions))]

        
        if self.inference_model.model_name == "kaist-ai/prometheus-8x7b-v2":
             # each prediction would be an tuple of two responses 
            instances = [{
                    "context_variables": input_instance['context'],
                    "response_a" : prediction[0],
                    "response_b" : prediction[1],
                    "option_a" : option_pair[0],
                    "option_b" : option_pair[1],
                    "rubric" : f"{pairwise_criteria.name}: {pairwise_criteria.criteria}",
                } for prediction, option_pair, input_instance in zip(predictions, option_pairs, task_data)]
            
            # Assessment stage
            assessment_outputs = infer(instances, task=self.assessment_task_prometheus, engine=self.inference_model, template=self.assessment_template)
            self.logger.info("generated assessment")

            feedbacks = [out.split("[RESULT]")[0].strip() for out in assessment_outputs]
            return [{"score": 0.8, "assessment": assessment_outputs[i], "summary": feedbacks[i]} for i in range(len(predictions))]
        
        else:   
            # each prediction would be an tuple of two responses 
            instances = [{
                    "context_variables": input_instance['context'],
                    "response_a" : prediction[0],
                    "response_b" : prediction[1],
                    "option_a" : option_pair[0],
                    "option_b" : option_pair[1],
                    "criteria_name" : pairwise_criteria.name,
                    "criteria_description" : pairwise_criteria.criteria
                } for prediction, option_pair, input_instance in zip(predictions, option_pairs, task_data)]
            
            # Assessment stage
            assessment_outputs = infer(instances, task=self.assessment_task, engine=self.inference_model, template=self.assessment_template)
            self.logger.info("generated assessment")
            assessment_instances = [{"assessment": assessment_output} for assessment_output in assessment_outputs]

            # Summarization stage
            summ_output = infer(assessment_instances, task=self.summ_task, engine=self.inference_model, template=self.summ_template)
            self.logger.info("generated summary")

            return [{"score": 0.8, "assessment": assessment_outputs[i], "summary": summ_output[i]} for i in range(len(predictions))]
        