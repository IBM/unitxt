from ..inference import InferenceEngine, OpenAiInferenceEngine, WMLInferenceEngine
from ..operator import SequentialOperator
from ..rag_metrics.task_based_judge_metric import TaskBasedJudgeMetric
from ..stream import MultiStream


class GenerativeBinaryJudge(TaskBasedJudgeMetric):
    inference_engine_class = None
    inference_model: InferenceEngine = None
    generation_kwargs: dict = {}

    def prepare(self):
        super().prepare()
        if self.inference_model is None:
            self.inference_model = self.inference_engine_class(
                model_name=self.model_name,
                **self.generation_kwargs,
            )

        # the processing steps for converting the logprobs dicts to float predictions
        self.post_processor = SequentialOperator(
            steps=[
                "processors.infer_logprobs_to_yes_no_probs",
                "processors.cast_to_float_return_zero_if_failed",
            ]
        )

    def compute(
        self,
        references: list[list[str]],
        predictions: list[str],
        task_data: list[dict],
    ) -> dict:
        processed_data = self._prepare_instances_for_model(task_data)
        preds = self.inference_model.infer_log_probs(processed_data)
        processed_preds = self._process_predictions(preds)

        return [{self.main_score: s} for s in processed_preds]

    def _prepare_instances_for_model(self, task_data: list[dict]):
        task_data = self.adjust_instances_to_task(task_data)
        stream = MultiStream({"test": task_data})

        processed_stream = self.processor.process(stream)
        return processed_stream.to_dataset()["test"]

    def _process_predictions(self, logprob_results: list[dict]):
        stream = MultiStream(
            {
                "test": [
                    {
                        "prediction": x,
                        "references": [0.0],
                    }  # the format expected by prediction post-processors
                    for x in logprob_results
                ]
            }
        )
        processed_stream = self.post_processor.process(stream)
        return processed_stream.to_dataset()["test"]["prediction"]


class GenerativeBinaryJudgeWML(GenerativeBinaryJudge):
    inference_engine_class = WMLInferenceEngine
    generation_kwargs = {"max_new_tokens": 5}

    _requirements_list: list[str] = ["ibm_watsonx_ai"]


class GenerativeBinaryJudgeOpenAi(GenerativeBinaryJudge):
    inference_engine_class = OpenAiInferenceEngine
    generation_kwargs = {"logprobs": True, "max_tokens": 5}

    _requirements_list: list[str] = ["openai"]
