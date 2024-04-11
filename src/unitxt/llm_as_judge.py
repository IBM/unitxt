import ast
from typing import Any, Dict, List

from unitxt import produce
from unitxt.inference import InferenceEngine, PipelineBasedInferenceEngine
from unitxt.metrics import BulkInstanceMetric


class LLMAsJudge(BulkInstanceMetric):
    """LLM as judge based metric class for evaluating correctness.

    Attributes:
        main_score (str): The main score used for evaluation.
        reduction_map (dict): A dictionary specifying the reduction method for the metric.
        betch_size (int): The size of the bulk.
        recipe (str): The unitxt recipe that will be used to create the judge dataset.
        inference (InferenceEngine): the module that creates the inference.

    Methods:
        prepare(self): Initialization method for the metric.
        compute(self, references, predictions, additional_inputs): Method to compute the metric.

    Usage:
        metric = LlamaIndexCorrectnessMetric()
        scores = metric.compute(references, prediction, additional_inputs)
    """
    main_score: str = f"llm_as_judge"
    reduction_map: Dict[str, List[str]] = {"mean": [main_score]}
    batch_size: int = 32
    recipe: str = "card=cards.llm_as_judge.model_response_assessment.mt_bench," \
                  "template=templates.llm_as_judge.model_response_assessment.mt_bench," \
                  "demos_pool_size=0," \
                  "num_demos=0"

    inference_model: InferenceEngine = None

    def prepare(self):
        super().prepare()
        if self.inference_model is None:
            self.inference_model = PipelineBasedInferenceEngine(model_name='google/flan-t5-base', max_new_tokens=32)

    def compute(
            self,
            references: List[List[Any]],
            predictions: List[Any],
            task_data: List[Dict],
    ) -> List[Dict[str, Any]]:
        instances = [{**task_data_instance, **{'prediction': prediction, 'output': ''}}
                     for task_data_instance, prediction in zip(task_data, predictions)]

        dataset = produce(instances, self.recipe)
        verdicts = self.inference_model.infer(dataset)
        # TODO: Convert the verdicts to scores in a standard way (postprocessors?), It should be defined in the recipe (in the card?)
        eval_verdicts  = [ast.literal_eval(verdict) for verdict in verdicts]
        return [{self.main_score: verdict[0][0] / 10.0} for verdict in eval_verdicts]
