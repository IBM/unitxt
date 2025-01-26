from typing import Any

from .artifact import fetch_artifact
from .llm_as_judge_constants import Criteria, CriteriaOption, CriteriaWithOptions
from .operators import FieldOperator


class LoadCriteriaWithOptions(FieldOperator):
    def process_value(self, text: Any) -> CriteriaWithOptions:
        return fetch_artifact(text)[0]


class CreateCriteriaWithOptionsFromDict(FieldOperator):
    def process_value(self, criteria_dict: dict) -> Any:
        return CriteriaWithOptions.from_obj(criteria_dict)


class CreateCriteriaWithOptionsFromJson(FieldOperator):
    def process_value(self, text: str) -> Any:
        return CriteriaWithOptions.from_jsons(text)


class CreateYesNoCriteriaFromString(FieldOperator):
    def process_value(self, text: Any) -> Any:
        return CriteriaWithOptions(
            name="",
            description=text,
            options=[
                CriteriaOption(name="Yes", description=""),
                CriteriaOption(name="No", description=""),
            ],
            option_map={
                "Yes": 1.0,
                "No": 0.0,
            },
        )


class CreateYesNoPartiallyCriteriaFromString(FieldOperator):
    def process_value(self, text: str) -> Any:
        return CriteriaWithOptions(
            name="",
            description=text,
            options=[
                CriteriaOption(name="Yes", description=""),
                CriteriaOption(name="Partially", description=""),
                CriteriaOption(name="No", description=""),
            ],
            option_map={
                "Yes": 1.0,
                "Partially": 0.5,
                "No": 0.0,
            },
        )


class LoadCriteria(FieldOperator):
    def process_value(self, text: Any) -> Criteria:
        return fetch_artifact(text)[0]


class CreateCriteriaFromDict(FieldOperator):
    def process_value(self, criteria_dict: dict) -> Any:
        return Criteria.from_obj(criteria_dict)


class CreateCriteriaFromJson(FieldOperator):
    def process_value(self, text: str) -> Any:
        return Criteria.from_jsons(text)


class CreateCriteriaFromString(FieldOperator):
    def process_value(self, text: str) -> Any:
        return Criteria(
            name="",
            description=text,
        )
