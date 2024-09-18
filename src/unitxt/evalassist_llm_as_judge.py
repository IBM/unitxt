from .metrics import BulkInstanceMetric
from .inference import InferenceEngine
from .templates import Template, InputOutputTemplate
from .task import Task
from .api import infer

from typing import List,Dict

class RubricOption:
    def __init__(self, option: str, description: str):
        self.option = option
        self.description = description

class EvalAssistLLMAsJudge(BulkInstanceMetric):
    inference_model: InferenceEngine
    # template: Template
    reduction_map = {"mean": ["score"]}
    main_score = "score"
    print("------------inside eval assist llm as a judge-------------")
   
    def prepare(self):
        print("------------inside prepare of eval assist llm as a judge-------------")
        self.assessment_template = """<s> [INST]
You are presented with a response generated subject to a context.
The context includes information relevant to the nature or generation of the response.
You will assess the quality of the response subject to an evaluation criteria.

###Context:
{context_variables}

###Response:
{response}

###Evaluation criteria:
{criteria}
{options}

Briefly assess the quality of the response subject to the evaluation criteria.
Focus on the evaluation criteria during assessment, do not provide a general assessment.
[/INST]
Assessment:
"""

        self.summarization_prompt = """<s> [INST]
Summarize the following assessment while maintaining the pertinent details.
Assessment: {assessment}
[/INST]
Assessment Summary:
"""
        self.answer_prompt = """</s>
[INST]
Now consider the evaluation criteria and choose a final answer.
Validate the answer against the assessment.
###Evaluation criteria:
{{criteria}}
{{options}}
[/INST]
Answer:"""

        self.rubric_options = [RubricOption("Yes", "The temperature reading is provided in both Fahrenheit and Celsius."),
                            RubricOption("No", "The temperature reading is provided either in Fahrenheit or Celsius, but not both."),
                             RubricOption("Pass", "There is no numerical temperature reading in the response.")]
        self.options_as_str = "Choose an answer:\n" + "\n".join([f"- \"{o.option}\"{f' if {o.description}' if o.description != '' else ''}" for o in self.rubric_options])
        self.criteria = "In the response, if there is a numerical temperature present, is it denominated in both Fahrenheit and Celsius?"
        super().prepare()

    def compute(
        self, references: List[List[str]], predictions: List[str], task_data: List[Dict[str,str]]
    ) -> dict:
        instances = [{
                    "context_variables": input_instance['question'],
                    "response": prediction,
                    "criteria": criterion,
                    "options": option
                }
                for input_instance, prediction, criterion, option in zip(
                    task_data, predictions, [self.criteria] * len(predictions), [self.options_as_str] * len(predictions))]

        assessment_template = InputOutputTemplate(input_format=self.assessment_template)
        summ_template = InputOutputTemplate(input_format=self.summarization_prompt)
        assessment_task =  Task(
            input_fields={"context_variables": str, "response": str, "criteria": str, "options": str},
            reference_fields={},
            prediction_type=str,
            metrics=[])
        assessment_outputs = infer(instances, task=assessment_task, engine=self.inference_model, template=assessment_template)
        print("ass_output ", assessment_outputs)

        assessment_instances = [{"assessment": assessment_output} for assessment_output in assessment_outputs]
        summ_task =  Task(
            input_fields={"assessment": str},
            reference_fields={},
            prediction_type=str,
            metrics=[])
        summ_output = infer(assessment_instances, task=summ_task, engine=self.inference_model, template=summ_template)
        print("summ_output ", summ_output)

        return [{"score": 0.8, "assessment": assessment_outputs[i], "summary": summ_output[i]} for i in range(len(predictions))]