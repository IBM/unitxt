import unittest

from src.unitxt.operators import (
    MapInstanceValues,
    FlattenInstances,
    FilterByValues,
    ApplyValueOperatorsField,
    Unique,
)

from src.unitxt.test_utils.operators import apply_operator

class TestOperators(unittest.TestCase):
    
    def test_map_instance_values(self):
        
        inputs = [
            {'a': 1, 'b': 2},
            {'a': 2, 'b': 3},
        ]
        
        targets = [
            {'a': 'hi', 'b': 2},
            {'a': 'bye', 'b': 3},
        ]
        
        outputs = apply_operator(
            operator=MapInstanceValues(mappers={'a': {'1': 'hi', '2': 'bye'}}),
            inputs=inputs
        )
        
        for output, target in zip(outputs, targets):
            self.assertDictEqual(output, target)

    
    def test_flatten_instances(self):
        
        inputs = [
            {'a': {'b': 1}},
            {'a': {'b': 2}},
        ]
        
        targets = [
            {'a...b': 1},
            {'a...b': 2},
        ]
        
        outputs = apply_operator(
            operator=FlattenInstances(sep='...'),
            inputs=inputs
        )
        
        for output, target in zip(outputs, targets):
            self.assertDictEqual(output, target)
        
    