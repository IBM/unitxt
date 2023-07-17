import unittest

from src.unitxt.operators import (
    MapInstanceValues,
    FlattenInstances,
    FilterByValues,
    ApplyValueOperatorsField,
    Unique,
)

from src.unitxt.test_utils.operators import test_operator

class TestOperators(unittest.TestCase):
    
    def test_map_instance_values(self):
        
        inputs = [
            {'a': 1, 'b': 2},
            {'a': 2, 'b': 3},
        ]
        
        outputs = [
            {'a': 'hi', 'b': 2},
            {'a': 'bye', 'b': 3},
        ]
        
        result = test_operator(
            MapInstanceValues(mappers={'a': {'1': 'hi', '2': 'bye'}}),
            inputs, outputs
        )
        
        self.assertTrue(result)
        
    