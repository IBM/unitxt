import unittest

from src.unitxt.operators import (
    AddFields,
    ApplyOperatorsField,
    AugmentWhitespace,
    CastFields,
    CopyFields,
    DeterministicBalancer,
    EncodeLabels,
    FilterByValues,
    FlattenInstances,
    JoinStr,
    LengthBalancer,
    MapInstanceValues,
    RemoveFields,
    RenameFields,
    Shuffle,
    SplitByValue,
    StreamRefiner,
    TakeByField,
    Unique,
    ZipFieldValues,
)
from src.unitxt.stream import MultiStream, Stream
from src.unitxt.test_utils.operators import apply_operator, test_operator


class TestOperators(unittest.TestCase):
    def test_map_instance_values(self):
        inputs = [
            {"a": 1, "b": 2},
            {"a": 2, "b": 3},
        ]

        targets = [
            {"a": "hi", "b": 2},
            {"a": "bye", "b": 3},
        ]

        test_operator(
            operator=MapInstanceValues(mappers={"a": {"1": "hi", "2": "bye"}}),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_flatten_instances(self):
        inputs = [
            {"a": {"b": 1}},
            {"a": {"b": 2}},
        ]

        targets = [
            {"a...b": 1},
            {"a...b": 2},
        ]

        test_operator(operator=FlattenInstances(sep="..."), inputs=inputs, targets=targets, tester=self)

    def test_filter_by_values(self):
        inputs = [
            {"a": 1, "b": 2},
            {"a": 2, "b": 3},
        ]

        targets = [
            {"a": 1, "b": 2},
        ]

        test_operator(operator=FilterByValues(values={"a": 1}), inputs=inputs, targets=targets, tester=self)

    def test_apply_value_operators_field(self):
        inputs = [
            {"a": 111, "b": 2, "c": "processors.to_string"},
            {"a": 222, "b": 3, "c": "processors.to_string"},
        ]

        targets = [
            {"a": "111", "b": 2, "c": "processors.to_string"},
            {"a": "222", "b": 3, "c": "processors.to_string"},
        ]

        test_operator(
            operator=ApplyOperatorsField(inputs_fields=["a"], operators_field="c", default_operators=["add"]),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_add_fields(self):
        inputs = [
            {"a": 1, "b": 2},
            {"a": 2, "b": 3},
        ]

        targets = [
            {"a": 1, "b": 2, "c": 3},
            {"a": 2, "b": 3, "c": 3},
        ]

        test_operator(operator=AddFields(fields={"c": 3}), inputs=inputs, targets=targets, tester=self)

    def test_remove_fields(self):
        inputs = [
            {"a": 1, "b": 2},
            {"a": 2, "b": 3},
        ]

        targets = [
            {"a": 1},
            {"a": 2},
        ]

        test_operator(operator=RemoveFields(fields=["b"]), inputs=inputs, targets=targets, tester=self)

    def test_unique_on_single_field(self):
        inputs = [
            {"a": 1, "b": 2},
            {"a": 2, "b": 3},
            {"a": 2, "b": 4},
        ]

        targets = {(1,), (2,)}

        outputs = apply_operator(
            operator=Unique(fields=["a"]),
            inputs=inputs,
        )

        self.assertSetEqual(set(outputs), targets)

    def test_unique_on_multiple_fields(self):
        inputs = [
            {"a": 1, "b": 2},
            {"a": 2, "b": 3},
            {"a": 2, "b": 4},
            {"a": 1, "b": 2},
        ]
        fields = ["a", "b"]
        targets = {(1, 2), (2, 3), (2, 4)}

        outputs = apply_operator(
            operator=Unique(fields=fields),
            inputs=inputs,
        )

        self.assertSetEqual(set(outputs), targets)

    def test_split_by_value(self):
        inputs = [
            {"a": 1, "b": 4},
            {"a": 2, "b": 3},
            {"a": 2, "b": 4},
        ]

        outputs = apply_operator(operator=SplitByValue(fields="a"), inputs=inputs, return_multi_stream=True)

        self.assertSetEqual(set(outputs.keys()), {"test_1", "test_2"})

        outputs_1 = list(outputs["test_1"])
        self.assertEqual(len(outputs_1), 1)

        outputs_2 = list(outputs["test_2"])
        self.assertEqual(len(outputs_2), 2)

        for input_dict, ouput_dict in zip(inputs, outputs_1):
            self.assertDictEqual(input_dict, ouput_dict)

        for input_dict, ouput_dict in zip(inputs[1:], outputs_2):
            self.assertDictEqual(input_dict, ouput_dict)

    def test_shuffle(self):
        inputs = [{"a": i} for i in range(15)]

        outputs = apply_operator(operator=Shuffle(page_size=10), inputs=inputs)

        inputs = [instance["a"] for instance in inputs]
        outputs = [instance["a"] for instance in outputs]

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
            {"a": "0.5", "b": "2"},
            {"a": "fail", "b": "fail"},
        ]

        targets = [
            {"a": 0.5, "b": 2},
            {"a": 0.0, "b": 0},
        ]

        test_operator(
            operator=CastFields(fields={"a": "float", "b": "int"}, failure_defaults={"a": 0.0, "b": 0}),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_test_cast_fields_casting_failure(self):
        inputs = [
            {"a": "0.5", "b": "2"},
            {"a": "fail", "b": "fail"},
        ]

        with self.assertRaises(ValueError):
            outputs = apply_operator(operator=CastFields(fields={"a": "float", "b": "int"}), inputs=inputs)

    def test_rename_fields(self):
        inputs = [
            {"a": 1, "b": 2},
            {"a": 2, "b": 3},
        ]

        targets = [
            {"a": 1, "c": 2},
            {"a": 2, "c": 3},
        ]

        test_operator(operator=RenameFields(field_to_field={"b": "c"}), inputs=inputs, targets=targets, tester=self)

    def test_copy_paste_fields(self):
        inputs = [
            {"a": [1, 3]},
            {"a": [2, 4]},
        ]

        targets = [{"a": 1}, {"a": 2}]

        test_operator(
            operator=CopyFields(field_to_field={"a/0": "a"}, use_query=True),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_copy_paste_same_name2(self):
        inputs = [
            {"a": "test"},
            {"a": "pest"},
        ]

        targets = [{"a": {"x": "test"}}, {"a": {"x": "pest"}}]

        test_operator(
            operator=CopyFields(field_to_field={"a": "a/x"}, use_query=True),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_label_encoder(self):
        inputs = [
            {"prediction": "red", "references": ["red", "blue"]},
            {"prediction": "blue", "references": ["blue"]},
            {"prediction": "green", "references": ["red"]},
        ]

        targets = [
            {"prediction": 0, "references": [0, 1]},
            {"prediction": 1, "references": [1]},
            {"prediction": 2, "references": [0]},
        ]

        test_operator(
            operator=EncodeLabels(fields=["prediction", "references/*"]), inputs=inputs, targets=targets, tester=self
        )

    def test_join_str(self):
        inputs = [
            {"a": [1, 3]},
            {"a": [2, 4]},
        ]

        targets = [
            {"a": [1, 3], "b": "1,3"},
            {"a": [2, 4], "b": "2,4"},
        ]

        test_operator(
            operator=JoinStr(field_to_field={"a": "b"}, separator=","), inputs=inputs, targets=targets, tester=self
        )

    def test_zip_fields(self):
        inputs = [
            {"a": [1, 3], "b": [1, 3]},
            {"a": [2, 4], "b": [2, 4]},
        ]

        targets = [
            {"a": [1, 3], "b": [1, 3], "c": [(1, 1), (3, 3)]},
            {"a": [2, 4], "b": [2, 4], "c": [(2, 2), (4, 4)]},
        ]

        test_operator(
            operator=ZipFieldValues(fields=["a", "b"], to_field="c", use_query=True),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_take_by_field(self):
        inputs = [
            {"a": [1, 3], "b": 0},
            {"a": {"a": 1}, "b": "a"},
        ]

        targets = [
            {"a": [1, 3], "b": 0, "c": 1},
            {"a": {"a": 1}, "b": "a", "c": 1},
        ]

        test_operator(
            operator=TakeByField(field="a", index="b", to_field="c", use_query=True),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_stream_refiner(self):
        refiner = StreamRefiner()

        ms = MultiStream.from_iterables({"train": [{"x": 0}, {"x": 1}], "test": [{"x": 2}, {"x": 3}]}, copying=True)

        refiner.apply_to_streams = ["train"]
        refiner.max_instances = 1

        refined_ms = refiner(ms)

        train = list(refined_ms["train"])
        self.assertEqual(len(train), 1)

        test = list(refined_ms["test"])
        self.assertEqual(len(test), 2)

    def test_deterministic_balancer(self):
        inputs = [
            {"a": [1, 3], "b": 0, "id": 0},
            {"a": [1, 3], "b": 0, "id": 1},
            {"a": {"a": 1}, "b": "a", "id": 2},
        ]

        targets = [
            {"a": [1, 3], "b": 0, "id": 0},
            {"a": {"a": 1}, "b": "a", "id": 2},
        ]

        test_operator(
            operator=DeterministicBalancer(fields=["a", "b"]),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_length_balancer(self):
        inputs = [
            {"a": [1, 3], "b": 0, "id": 0},
            {"a": [1, 3], "b": 0, "id": 1},
            {"a": [], "b": "a", "id": 2},
        ]

        targets = [
            {"a": [1, 3], "b": 0, "id": 0},
            {"a": [], "b": "a", "id": 2},
        ]

        test_operator(
            operator=LengthBalancer(fields=["a"], segments_boundaries=[1]),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_augment_whitespace(self):
        source = "The dog ate my cat"
        inputs = [{"source": source}]

        operator = AugmentWhitespace()
        outputs = apply_operator(operator, inputs)
        assert outputs[0]["source"] != source, f"Source of f{outputs} is equal to f{source} and was not augmented"
        normalized_output_source = outputs[0]["source"].split()
        normalized_input_source = source.split()
        assert (
            normalized_output_source == normalized_input_source
        ), f"{normalized_output_source} is not equal to f{normalized_input_source}"
