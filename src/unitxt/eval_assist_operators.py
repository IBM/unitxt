import json
from typing import Any

from .artifact import fetch_artifact
from .eval_assist_constants import CriteriaOption, CriteriaWithOptions
from .operators import FieldOperator


class LoadCriteria(FieldOperator):
    def process_value(self, text: Any) -> Any:
        return fetch_artifact(text)[0]


class CreateCriteriaFromDict(FieldOperator):
    def process_value(self, text: Any) -> Any:
        return CriteriaWithOptions(
            name=text["name"],
            description=text["description"],
            options=[
                CriteriaOption(
                    name=option_dict["name"],
                    description=option_dict["description"],
                )
                for option_dict in text["options"]
            ],
        )


class CreateCriteriaFromJson(CreateCriteriaFromDict):
    def process_value(self, text: Any) -> Any:
        dict = json.loads(text)
        return super().process_value(dict)


class CreateYesNoCriteriaFromString(FieldOperator):
    def process_value(self, text: Any) -> Any:
        return CriteriaWithOptions(
            name=f"Unknown ({text[:20]}...)",
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
