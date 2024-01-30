from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class InstanceInput(BaseModel):
    prediction: Any
    references: List[Any]
    additional_inputs: Optional[Dict] = None


class InstanceOutput(InstanceInput):
    instance_scores: Dict[str, Any]


class MetricRequest(BaseModel):
    instance_inputs: List[InstanceInput]


class MetricResponse(BaseModel):
    instance_outputs: List[InstanceOutput]
    global_score: Dict[str, Any]
