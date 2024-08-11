from typing import Dict, List

from ..catalog import get_from_catalog
from ..metrics import BulkInstanceMetric
from ..operator import SequentialOperator


class TaskBasedJudgeMetric(BulkInstanceMetric):
    reduction_map: Dict[str, List[str]] = None
    prediction_type = float
    model_name: str = None
    task_name: str = None
    template_name: str = None
    model_format_name = "formats.empty"

    def prepare(
        self,
    ):
        self.reduction_map = {"mean": [self.main_score]}
        # the processing steps for preparing the prompt (instruction, answer prefix etc.)
        # that we send to the generative judge model
        self.processor = SequentialOperator(
            steps=[
                self.task_name,
                self.template_name,
                self.model_format_name,
            ]
        )
        self.set_unneeded_fields()

    def compute(
        self,
        references: list[list[str]],
        predictions: list[str],
        task_data: list[dict],
    ) -> dict:
        pass

    def set_unneeded_fields(self):
        template_input_format = get_from_catalog(self.template_name).input_format
        task = get_from_catalog(self.task_name)
        unneeded_task_fields = {
            **task.reference_fields,
            **{
                field: field_type
                for field, field_type in task.input_fields.items()
                if field not in template_input_format
            },
        }
        self.unneeded_task_fields = unneeded_task_fields

    def adjust_instances_to_task(self, task_data):
        # we add mock values for fields that the task expects but are not needed here
        for field, field_type in self.unneeded_task_fields.items():
            if field not in task_data[0]:
                for example in task_data:
                    example[field] = (
                        ["mock"] if field_type._name == "List" else 0.0
                    )  # this is a very specific hack
        return task_data
