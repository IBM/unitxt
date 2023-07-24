import unittest

from src.unitxt.operators import (
    MapInstanceValues,
    FlattenInstances,
    FilterByValues,
    ApplyValueOperatorsField,
    AddFields,
    Unique,
    Shuffle,
    CastFields,
    CopyPasteFields,
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
    
    def test_shuffle(self):
        
        inputs = [{'a': i} for i in range(15)]
        
        outputs = apply_operator(
            operator=Shuffle(page_size=10),
            inputs=inputs
        )
        
        inputs = [instance['a'] for instance in inputs]
        outputs = [instance['a'] for instance in outputs]
        
        self.assertNotEqual(inputs, outputs)
        self.assertSetEqual(set(inputs), set(outputs))
        
        # test no mixing between pages:
        page_1_inputs = inputs[:10]
        page_2_inputs = inputs[10:]
        page_1_outputs = outputs[:10]
        page_2_outputs = outputs[10:]
        
        self.assertSetEqual(set(page_1_inputs), set(page_1_outputs))
        self.assertSetEqual(set(page_2_inputs), set(page_2_outputs))
        
        inputs_outputs_intersection = set(page_1_inputs).intersection(set(page_2_outputs))
        self.assertSetEqual(inputs_outputs_intersection, set())
        
        inputs_outputs_intersection = set(page_2_inputs).intersection(set(page_1_outputs))
        self.assertSetEqual(inputs_outputs_intersection, set())

    def test_cast_fields(self):
        
        inputs = [
            {'a': '0.5', 'b': '2'},
            {'a': 'fail', 'b': 'fail'},
        ]
        
        targets = [
            {'a': 0.5, 'b': 2},
            {'a': 0.0, 'b': 0},
        ]
        
        outputs = apply_operator(
            operator=CastFields(fields={'a': 'float', 'b': 'int'}, failure_defaults={'a': 0.0, 'b': 0}),
            inputs=inputs
        )
        
        for output, target in zip(outputs, targets):
            self.assertDictEqual(output, target)
    
    def test_test_cast_fields_casting_failure(self):
         
        inputs = [
            {'a': '0.5', 'b': '2'},
            {'a': 'fail', 'b': 'fail'},
        ]

        with self.assertRaises(ValueError):
            outputs = apply_operator(
                operator=CastFields(fields={'a': 'float', 'b': 'int'}),
                inputs=inputs
            )
    
            
    def test_copy_paste_fields(self):
        
        inputs = [
            {'a': 1, 'b': 2},
            {'a': 2, 'b': 3},
        ]
        
        targets = [
            {'a': 1, 'b': 2, 'c': 2},
            {'a': 2, 'b': 3, 'c': 3},
        ]
        
        outputs = apply_operator(
            operator=CopyPasteFields({'b': 'c'}),
            inputs=inputs
        )
        
        for output, target in zip(outputs, targets):
            self.assertDictEqual(output, target)
        
    
    def test_copy_patse_same_name(self):
        
        inputs = [
            {'a': [1, 3]},
            {'a': [2, 4]},
        ]
        
        targets = [
            {'a': 1},
            {'a': 2}
        ]
        
        outputs = apply_operator(
            operator=CopyPasteFields({'a/0': 'a'}, use_nested_query=True),
            inputs=inputs
        )
        
        for output, target in zip(outputs, targets):
            self.assertDictEqual(output, target)
            
    
    def test_copy_patse_same_name2(self):
        
        inputs = [
            {'a': 'test'},
            {'a': 'pest'},
        ]
        
        targets = [
            {'a': {'x': 'test'}},
            {'a': {'x': 'pest'}}
        ]
        
        outputs = apply_operator(
            operator=CopyPasteFields({'a': 'a/x'}, use_nested_query=True),
            inputs=inputs
        )
        
        for output, target in zip(outputs, targets):
            self.assertDictEqual(output, target)