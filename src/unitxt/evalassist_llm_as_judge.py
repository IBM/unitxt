from .metrics import BulkInstanceMetric, InstanceMetric
from .inference import InferenceEngine
from .templates import Template
from jinja2 import Environment, FileSystemLoader

from typing import List,Dict

class RubricOption:
    def __init__(self, option: str, description: str):
        self.option = option
        self.description = description

class EvalAssistLLMAsJudge(BulkInstanceMetric):
    inference_model: InferenceEngine
    # template: Template
    prompt_template_path: str 
    main_score: str = "eval_assist_llm_as_judge"
    reduction_map = {"mean": ["score"]}
    main_score = "score"

    print("------------inside eval assist llm as a judge-------------")
   
    def prepare(self):
        print("self.template path is ", self.prompt_template_path)
        print("------------inside prepare of eval assist llm as a judge-------------")
        env = Environment(loader = FileSystemLoader(self.prompt_template_path))
        self.assessment_template = env.get_template('mixtral/assessment.jinja')
        self.summarization_template = env.get_template('mixtral/summarization.jinja')
        self.answer_template = env.get_template('mixtral/answer.jinja')
        super().prepare()
        
    def compute(
        self, references: List[List[str]], predictions: List[str], task_data: List[Dict[str,str]]
    ) -> dict:
        output = self.assessment_template.render(context_variables = 'Some context',
                                                  response="Some response", criteria="Some criteria", options="some options")
        print("rendered template ", output)

        outputs = self.inference_model.infer(task_data)
        print("!! outputs are ", outputs)
        return [{"score": 0.8, "value":outputs[0] }, {"score": 0.9, "value":outputs[1]},
                 {"score": 0.9, "value":outputs[2]}]