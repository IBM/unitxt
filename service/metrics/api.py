from typing import Any, Dict, List, Optional

from pydantic import BaseModel

"""
This module defines the API of a metric service.
A single input to the service is a MetricRequest. The response
is returned in a MetricResponse object.
"""


class InstanceInput(BaseModel):
    prediction: Any
    references: List[Any]
    additional_inputs: Optional[Dict] = None


class MetricRequest(BaseModel):
    instance_inputs: List[InstanceInput]


class MetricResponse(BaseModel):
    # A list of instance score dictionaries. Each dictionary contains the
    # score names and score values for a single instance.
    instances_scores: List[Dict[str, Any]]
    # The global scores dictionary, containing global score names and values.
    # These are scores computed over the entire set of input instances, e.g.
    # an average over a score computed per instance.
    global_score: Dict[str, Any]
