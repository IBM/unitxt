from abc import abstractmethod
from typing import Any, Dict, List, Optional

from .logging_utils import get_logger
from .metrics import MapReduceMetric

logger = get_logger(__name__)


class BaseLLMJudge(MapReduceMetric[Any, Dict[str, Any]]):
    """Base class for all LLM-as-Judge implementations with shared functionality.

    This class provides common map-reduce patterns, score aggregation, and confidence interval handling
    for all LLM judge implementations. It defines the standard evaluation workflow using a two-step
    process: instance preparation followed by batch inference execution.

    Args:
        ci_score_names: Names of scores for which confidence intervals should be computed.
            Defaults to None, which means no confidence intervals are calculated.
    """

    ci_score_names: Optional[List[str]] = None

    def map(
        self, prediction: Any, references: List[Any], task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Single instance processing - redirects to map_stream for batch efficiency."""
        raise NotImplementedError(
            "LLM judge metrics should override map_stream for efficient batch processing, not map"
        )

    def map_stream(self, evaluation_inputs_stream):
        """Common map_stream implementation for all LLM judge subclasses."""
        logger.info(
            f'Starting evaluation with {self.__class__.__name__} using "{self._get_engine_id()}"'
        )

        # Prepare all instances for inference without aggregating heavy data
        prepared_instances = []
        for prediction, references, task_data in evaluation_inputs_stream:
            prepared_instance = self._prepare_instance_for_inference(
                prediction, references, task_data
            )
            prepared_instances.append(prepared_instance)

        # Run all inference steps on the prepared instances
        return self._run_inference_on_all(prepared_instances)

    @abstractmethod
    def _prepare_instance_for_inference(self, prediction, references, task_data):
        """Prepare a single instance for inference without keeping heavy data.

        This method should be implemented by each judge subclass to prepare
        an individual instance for batch inference processing.
        """
        pass

    @abstractmethod
    def _run_inference_on_all(self, prepared_instances):
        """Run inference on all prepared instances efficiently.

        This method should be implemented by each judge subclass to execute
        inference on the batch of prepared instances and return results.
        """
        pass

    def reduce(self, intermediates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate individual instance results into global scores."""
        if not intermediates:
            return {}

        aggregated = {}

        # For LLM judges, only aggregate the main score field (like original BulkInstanceMetric behavior)
        if hasattr(self, "main_score") and self.main_score:
            # Collect values only for the main score field
            values = []
            for result in intermediates:
                if self.main_score in result and isinstance(
                    result[self.main_score], (int, float)
                ):
                    values.append(result[self.main_score])

            if values:
                aggregated[self.main_score] = sum(values) / len(values)
                # Set the score alias
                aggregated["score"] = aggregated[self.main_score]
                aggregated["score_name"] = self.main_score

        return aggregated

    def reduce_one(self, intermediate: Dict[str, Any]) -> Dict[str, Any]:
        """Return individual instance scores."""
        result = dict(intermediate)
        if (
            hasattr(self, "main_score")
            and self.main_score
            and self.main_score in result
        ):
            result["score"] = result[self.main_score]
            result["score_name"] = self.main_score
        return result

    def _get_engine_id(self):
        if hasattr(self, "inference_engine"):
            return self.inference_engine.get_engine_id()
        if hasattr(self, "inference_model"):
            return self.inference_model.get_engine_id()
        return "unknown_engine"
