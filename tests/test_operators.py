import unittest

from src.unitxt.operators import (
    MapInstanceValues,
    FlattenInstances,
    FilterByValues,
    ApplyValueOperatorsField,
    AddFields,
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
    
    def test_filter_by_values(self):
        
        inputs = [
            {'a': 1, 'b': 2},
            {'a': 2, 'b': 3},
        ]
        
        targets = [
            {'a': 1, 'b': 2},
        ]
        
        outputs = apply_operator(
            operator=FilterByValues(values={'a': 1}),
            inputs=inputs
        )
        
        for output, target in zip(outputs, targets):
            self.assertDictEqual(output, target)
        
    def test_apply_value_operators_field(self):
        
        inputs = [
            {'a': 111, 'b': 2, 'c': 'processors.to_string'},
            {'a': 222, 'b': 3, 'c': 'processors.to_string'},
        ]
        
        targets = [
            {'a': '111', 'b': 2, 'c': 'processors.to_string'},
            {'a': '222', 'b': 3, 'c': 'processors.to_string'},
        ]
        
        outputs = apply_operator(
            operator=ApplyValueOperatorsField(value_field='a', operators_field='c', default_operators=['add']),
            inputs=inputs
        )
        
        for output, target in zip(outputs, targets):
            self.assertDictEqual(output, target)
    
    def test_add_fields(self):
        
        inputs = [
            {'a': 1, 'b': 2},
            {'a': 2, 'b': 3},
        ]
        
        targets = [
            {'a': 1, 'b': 2, 'c': 3},
            {'a': 2, 'b': 3, 'c': 3},
        ]
        
        outputs = apply_operator(
            operator=AddFields(fields={'c': 3}),
            inputs=inputs
        )
        
        for output, target in zip(outputs, targets):
            self.assertDictEqual(output, target)
        
    