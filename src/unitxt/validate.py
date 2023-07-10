from .text_utils import split_words
from .stream import StreamInstanceOperator, InstanceOperatorWithGlobalAccess, Artifact

from datasets import Value, Features, Dataset, Sequence

from dataclasses import field
from typing import Dict, Any
from abc import ABC, abstractmethod

class Validator(ABC):
    pass

class ValidateSchema(Validator, StreamInstanceOperator):
    
    schema: Features = None
    
    def verify(self):
        assert isinstance(self.schema, Features), 'Schema must be an instance of Features'
        assert self.schema is not None, 'Schema must be specified'
    
    def verify_first_instance(self, instance):
        for field in self.standart_fields:
            assert field in instance, f'Field "{field}" is missing in the first instance'
    
    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        return instance

class StandardSchema(Features):
    
    def __init__(self):
        super().__init__({
            'source': Value('string'),
            'target': Value('string'),
            'references': Sequence(Value('string')),
            'metrics': Sequence(Value('string')),
            'parser': Value('string'),
            # 'group': Value('string'),
            # 'guidance': Value('string'),
        })

class ValidateStandartSchema:
    
    schema: Features = field(default_factory=StandardSchema)
    
    

