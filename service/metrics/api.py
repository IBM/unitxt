from typing import Any, Dict, List, Optional

from pydantic import BaseModel, RootModel
from typing_extensions import TypedDict


class MetricInput(BaseModel):
    prediction: Any
    references: List[Any]
    additional_inputs: Optional[Dict] = None


class MetricOutput(MetricInput):
    score: TypedDict(
        "MetricScores", {"instance": Dict[str, Any], "global": Dict[str, Any]}
    )


class MetricRequest(RootModel):
    root: List[MetricInput]


class MetricResponse(RootModel):
    root: List[MetricOutput]
