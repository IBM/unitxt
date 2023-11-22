import json
import unittest

from src.unitxt.operators import (
    AddFields,
    ApplyOperatorsField,
    AugmentWhitespace,
    CastFields,
    CopyFields,
    DeterministicBalancer,
    EncodeLabels,
    ExtractFieldValues,
    FilterByListsOfValues,
    FilterByValues,
    FlattenInstances,
    Intersect,
    JoinStr,
    LengthBalancer,
    MapInstanceMultiLabelValues,
    MapInstanceValues,
    MergeStreams,
    RemoveFields,
    RemoveValues,
    RenameFields,
    Shuffle,
    SplitByValue,
    StreamRefiner,
    TakeByField,
    Unique,
    ZipFieldValues,
)
from src.unitxt.stream import MultiStream, Stream
from src.unitxt.test_utils.operators import (
    apply_operator,
    test_operator,
    test_operator_exception,
)


class TestOperators(unittest.TestCase):
    def compare_streams(self, all, expected_all):
        self.assertEqual(len(all), len(expected_all))
        for input_dict, output_dict in zip(all, expected_all):
            self.assertDictEqual(input_dict, output_dict)

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

    def test_map_instance_multi_label_values(self):
        inputs_p_e_v = [
            {"a": [1, 2, 3, 4], "b": 2},
            {"a": 2, "b": 3},
        ]

        inputs_n_p_e_v = [
            {"a": [1, 2, 3, 4], "b": 2},
            {"a": 2, "b": 3},
        ]

        targets_p_e_v = [
            {"a": ["hi", "bye", 3, 4], "b": 2},
            {"a": "bye", "b": 3},
        ]

        targets_n_p_e_v = [
            {"a": [1, 2, 3, 4], "b": 2},
            {"a": "bye", "b": 3},
        ]

        test_operator(
            operator=MapInstanceMultiLabelValues(
                mappers={"a": {"1": "hi", "2": "bye"}}, process_every_value=True, strict=False
            ),
            inputs=inputs_p_e_v,
            targets=targets_p_e_v,
            tester=self,
        )

        with self.assertRaises(ValueError) as cm:
            test_operator(
                operator=MapInstanceMultiLabelValues(
                    mappers={"a": {"hi": "ciao", "2": "bye", "3": "seeya"}}, process_every_value=True, strict=True
                ),
                inputs=inputs_p_e_v,
                targets=targets_p_e_v,
                tester=self,
            )

        test_operator(
            operator=MapInstanceMultiLabelValues(
                mappers={"a": {"1": "hi", "2": "bye"}}, process_every_value=False, strict=False
            ),
            inputs=inputs_n_p_e_v,
            targets=targets_n_p_e_v,
            tester=self,
        )

        with self.assertRaises(ValueError) as cm:
            test_operator(
                operator=MapInstanceMultiLabelValues(
                    mappers={"a": {"1": "hi", "2": "bye"}, "b": {"2": "ciao"}}, process_every_value=False, strict=True
                ),
                inputs=inputs_n_p_e_v,
                targets=targets_n_p_e_v,
                tester=self,
            )

    def test_map_instance_values_without_tester(self):
        inputs = [
            {"a": 1, "b": 2},
            {"a": 2, "b": 3},
        ]

        targets = [
            {"a": "hi", "b": 2},
            {"a": "bye", "b": 3},
        ]

        test_operator(
            operator=MapInstanceValues(mappers={"a": {"1": "hi", "2": "bye"}}), inputs=inputs, targets=targets
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
        inputs = [{"a": 1, "b": 2}, {"a": 2, "b": 3}, {"a": 1, "b": 3}]

        targets = [
            {"a": 1, "b": 3},
        ]

        test_operator(
            operator=FilterByValues(required_values={"a": 1, "b": 3}), inputs=inputs, targets=targets, tester=self
        )

        exception_text = "Required filter field ('c') in FilterByValues is not found in {'a': 1, 'b': 2}"
        test_operator_exception(
            operator=FilterByValues(required_values={"c": "5"}),
            inputs=inputs,
            exception_text=exception_text,
            tester=self,
        )

    def test_filter_by_list_of_values(self):
        inputs = [
            {"a": 1, "b": 2},
            {"a": 2, "b": 3},
            {"a": 3, "b": 4},
        ]

        targets = [
            {"a": 2, "b": 3},
            {"a": 3, "b": 4},
        ]

        test_operator(
            operator=FilterByListsOfValues(required_values={"b": ["3", "4"]}),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

        with self.assertRaises(ValueError) as cm:
            test_operator(
                operator=FilterByListsOfValues(required_values={"b": "5"}), inputs=inputs, targets=targets, tester=self
            )
        self.assertEqual(str(cm.exception), "The filter for key ('b') in FilterByListsOfValues is not a list but '5'")

        with self.assertRaises(ValueError) as cm:
            test_operator(
                operator=FilterByListsOfValues(required_values={"c": ["5"]}),
                inputs=inputs,
                targets=targets,
                tester=self,
            )
        self.assertEqual(
            str(cm.exception), "Required filter field ('c') in FilterByListsOfValues is not found in {'a': 1, 'b': 2}"
        )

    def test_intersect(self):
        inputs = [
            {"label": ["a", "b"]},
            {"label": ["a", "c", "d"]},
            {"label": ["a", "b", "f"]},
        ]

        targets = [
            {"label": ["b"]},
            {"label": []},
            {"label": ["b", "f"]},
        ]

        test_operator(
            operator=Intersect(field="label", allowed_values=["b", "f"]),
            inputs=inputs,
            targets=targets,
            tester=self,
        )
        with self.assertRaises(ValueError) as cm:
            test_operator(
                operator=Intersect(field="label", allowed_values=3), inputs=inputs, targets=targets, tester=self
            )
        self.assertEqual(str(cm.exception), "The allowed_values is not a list but '3'")

        with self.assertRaises(ValueError) as cm:
            test_operator(
                operator=Intersect(field="label", allowed_values=["3"], process_every_value=True),
                inputs=inputs,
                targets=targets,
                tester=self,
            )
        self.assertEqual(str(cm.exception), "'process_every_value=True' is not supported in Intersect operator")

        inputs = [
            {"label": "b"},
        ]
        exception_text = "Error processing instance '0' from stream 'test' in Intersect due to: Failed to process 'label' from {'label': 'b'} due to : The value in field is not a list but 'b'"
        test_operator_exception(
            operator=Intersect(field="label", allowed_values=["c"]),
            inputs=inputs,
            exception_text=exception_text,
            tester=self,
        )

    def test_remove_values(self):
        inputs = [
            {"label": ["a", "b"]},
            {"label": ["a", "c", "d"]},
            {"label": ["b", "f"]},
        ]

        targets = [
            {"label": ["a"]},
            {"label": ["a", "c", "d"]},
            {"label": []},
        ]

        test_operator(
            operator=RemoveValues(field="label", unallowed_values=["b", "f"]),
            inputs=inputs,
            targets=targets,
            tester=self,
        )
        with self.assertRaises(ValueError) as cm:
            test_operator(
                operator=RemoveValues(field="label", unallowed_values=3), inputs=inputs, targets=targets, tester=self
            )
        self.assertEqual(str(cm.exception), "The unallowed_values is not a list but '3'")

        with self.assertRaises(ValueError) as cm:
            test_operator(
                operator=RemoveValues(field="label", unallowed_values=["3"], process_every_value=True),
                inputs=inputs,
                targets=targets,
                tester=self,
            )
        self.assertEqual(str(cm.exception), "'process_every_value=True' is not supported in RemoveValues operator")

        inputs = [
            {"label": "b"},
        ]
        exception_text = "Error processing instance '0' from stream 'test' in RemoveValues due to: Failed to process 'label' from {'label': 'b'} due to : The value in field is not a list but 'b'"
        test_operator_exception(
            operator=RemoveValues(field="label", unallowed_values=["c"]),
            inputs=inputs,
            exception_text=exception_text,
            tester=self,
        )

        exception_text = "Error processing instance '0' from stream 'test' in RemoveValues due to: Failed to get 'label2' from {'label': 'b'} due to : query \"label2\" did not match any item in dict: {'label': 'b'}"
        test_operator_exception(
            operator=RemoveValues(field="label2", unallowed_values=["c"]),
            inputs=inputs,
            exception_text=exception_text,
            tester=self,
        )

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

    def test_merge(self):
        # Test with default params
        input_multi_stream = MultiStream(
            {"test": [{"field": "test1"}], "validation": [{"field": "validation1"}], "train": [{"field": "train1"}]}
        )
        output_multi_stream = MergeStreams()(input_multi_stream)
        self.assertListEqual(list(output_multi_stream.keys()), ["all"])
        all = list(output_multi_stream["all"])
        expected_all = [
            {"field": "test1", "origin": "test"},
            {"field": "validation1", "origin": "validation"},
            {"field": "train1", "origin": "train"},
        ]
        self.compare_streams(all, expected_all)

        # test with parameters
        input_multi_stream = MultiStream(
            {"test": [{"field": "test1"}], "validation": [{"field": "validation1"}], "train": [{"field": "train1"}]}
        )
        output_multi_stream = MergeStreams(
            streams_to_merge=["test", "train"], new_stream_name="merged", add_origin_stream_name=False
        )(input_multi_stream)
        self.assertListEqual(list(output_multi_stream.keys()), ["merged"])
        merged = list(output_multi_stream["merged"])
        expected_merged = [{"field": "test1"}, {"field": "train1"}]
        self.compare_streams(merged, expected_merged)

    def test_extract_values(self):
        input_multi_stream1 = MultiStream(
            {
                "test": [{"animal": "shark"}],
                "validation": [{"animal": "cat"}],
                "train": [
                    {"animal": "fish"},
                    {"animal": "dog"},
                    {"animal": "dog"},
                    {"animal": "cat"},
                    {"animal": "dog"},
                    {"animal": "cat"},
                    {"animal": "sheep"},
                    {"animal": "cat"},
                    {"animal": "fish"},
                    {"animal": "shark"},
                ],
            }
        )
        output_multi_stream = ExtractFieldValues(
            stream_name="train", field="animal", to_field="most_common_animals", overall_top_frequency_percent=80
        ).process(input_multi_stream1)
        expected_output1 = {
            "test": [{"animal": "shark", "most_common_animals": ["dog", "cat", "fish"]}],
            "validation": [{"animal": "cat", "most_common_animals": ["dog", "cat", "fish"]}],
            "train": [
                {"animal": "fish", "most_common_animals": ["dog", "cat", "fish"]},
                {"animal": "dog", "most_common_animals": ["dog", "cat", "fish"]},
                {"animal": "dog", "most_common_animals": ["dog", "cat", "fish"]},
                {"animal": "cat", "most_common_animals": ["dog", "cat", "fish"]},
                {"animal": "dog", "most_common_animals": ["dog", "cat", "fish"]},
                {"animal": "cat", "most_common_animals": ["dog", "cat", "fish"]},
                {"animal": "sheep", "most_common_animals": ["dog", "cat", "fish"]},
                {"animal": "cat", "most_common_animals": ["dog", "cat", "fish"]},
                {"animal": "fish", "most_common_animals": ["dog", "cat", "fish"]},
                {"animal": "shark", "most_common_animals": ["dog", "cat", "fish"]},
            ],
        }
        self.assertDictEqual(
            output_multi_stream,
            expected_output1,
            "expected to see: \n"
            + json.dumps(expected_output1)
            + "\n but instead, received: \n"
            + json.dumps(output_multi_stream),
        )
        # with minimum frequency limit
        output_multi_stream = ExtractFieldValues(
            stream_name="train",
            field="animal",
            to_field="most_common_animals",
            min_frequency_percent=25,
        ).process(input_multi_stream1)
        expected_output2 = {
            "test": [{"animal": "shark", "most_common_animals": ["dog", "cat"]}],
            "validation": [{"animal": "cat", "most_common_animals": ["dog", "cat"]}],
            "train": [
                {"animal": "fish", "most_common_animals": ["dog", "cat"]},
                {"animal": "dog", "most_common_animals": ["dog", "cat"]},
                {"animal": "dog", "most_common_animals": ["dog", "cat"]},
                {"animal": "cat", "most_common_animals": ["dog", "cat"]},
                {"animal": "dog", "most_common_animals": ["dog", "cat"]},
                {"animal": "cat", "most_common_animals": ["dog", "cat"]},
                {"animal": "sheep", "most_common_animals": ["dog", "cat"]},
                {"animal": "cat", "most_common_animals": ["dog", "cat"]},
                {"animal": "fish", "most_common_animals": ["dog", "cat"]},
                {"animal": "shark", "most_common_animals": ["dog", "cat"]},
            ],
        }
        self.assertDictEqual(
            output_multi_stream,
            expected_output2,
            "expected to see: \n"
            + json.dumps(expected_output2)
            + "\n but instead, received: \n"
            + json.dumps(output_multi_stream),
        )
        # with list values
        input_multi_stream2 = MultiStream(
            {
                "test": [{"field": ["a", "b", "c"]}],
                "validation": [{"field": ["d", "e", "f"]}],
                "train": [
                    # Individual value members and their overall frequency, in train:
                    # h: 6
                    # m: 6
                    # j: 3
                    # i: 3
                    # k: 3
                    # o: 3
                    # p: 3
                    # q: 2
                    # r: 2
                    # s: 2
                    # t: 1
                    # u:1
                    # v:1
                    # Tuples in train:
                    # ["h", "i", "j"] : 3
                    # ["k", "h", "m"]: 3
                    # ["m", "o", "p"] : 3
                    # ["q", "r", "s"] : 2
                    # ["t","u","v"] : 1
                    {"field": ["t", "u", "v"]},
                    {"field": ["h", "i", "j"]},
                    {"field": ["k", "h", "m"]},
                    {"field": ["m", "o", "p"]},
                    {"field": ["m", "o", "p"]},
                    {"field": ["h", "i", "j"]},
                    {"field": ["q", "r", "s"]},
                    {"field": ["k", "h", "m"]},
                    {"field": ["h", "i", "j"]},
                    {"field": ["q", "r", "s"]},
                    {"field": ["k", "h", "m"]},
                    {"field": ["m", "o", "p"]},
                ],
            }
        )
        # with lists, treated as single elements
        output_multi_stream = ExtractFieldValues(
            stream_name="train",
            field="field",
            to_field="most_common_lists",
            overall_top_frequency_percent=90,
            process_every_value=False,
        ).process(input_multi_stream2)

        expected_output3 = {
            "test": [
                {
                    "field": ["a", "b", "c"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                }
            ],
            "validation": [
                {
                    "field": ["d", "e", "f"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                }
            ],
            "train": [
                {
                    "field": ["t", "u", "v"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                },
                {
                    "field": ["h", "i", "j"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                },
                {
                    "field": ["k", "h", "m"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                },
                {
                    "field": ["m", "o", "p"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                },
                {
                    "field": ["m", "o", "p"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                },
                {
                    "field": ["h", "i", "j"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                },
                {
                    "field": ["q", "r", "s"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                },
                {
                    "field": ["k", "h", "m"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                },
                {
                    "field": ["h", "i", "j"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                },
                {
                    "field": ["q", "r", "s"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                },
                {
                    "field": ["k", "h", "m"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                },
                {
                    "field": ["m", "o", "p"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                },
            ],
        }

        self.assertDictEqual(
            output_multi_stream,
            expected_output3,
            "expected to see: \n"
            + json.dumps(expected_output3)
            + "\n but instead, received: \n"
            + json.dumps(output_multi_stream),
        )

        # finally, with lists and with process_every_value=True
        output_multi_stream = ExtractFieldValues(
            stream_name="train",
            field="field",
            to_field="most_common_individuals",
            overall_top_frequency_percent=90,
            process_every_value=True,
        ).process(input_multi_stream2)

        expected_output4 = {
            "test": [
                {
                    "field": ["a", "b", "c"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                    "most_common_individuals": ["h", "m", "i", "j", "k", "o", "p", "q", "r", "s"],
                }
            ],
            "validation": [
                {
                    "field": ["d", "e", "f"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                    "most_common_individuals": ["h", "m", "i", "j", "k", "o", "p", "q", "r", "s"],
                }
            ],
            "train": [
                {
                    "field": ["t", "u", "v"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                    "most_common_individuals": ["h", "m", "i", "j", "k", "o", "p", "q", "r", "s"],
                },
                {
                    "field": ["h", "i", "j"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                    "most_common_individuals": ["h", "m", "i", "j", "k", "o", "p", "q", "r", "s"],
                },
                {
                    "field": ["k", "h", "m"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                    "most_common_individuals": ["h", "m", "i", "j", "k", "o", "p", "q", "r", "s"],
                },
                {
                    "field": ["m", "o", "p"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                    "most_common_individuals": ["h", "m", "i", "j", "k", "o", "p", "q", "r", "s"],
                },
                {
                    "field": ["m", "o", "p"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                    "most_common_individuals": ["h", "m", "i", "j", "k", "o", "p", "q", "r", "s"],
                },
                {
                    "field": ["h", "i", "j"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                    "most_common_individuals": ["h", "m", "i", "j", "k", "o", "p", "q", "r", "s"],
                },
                {
                    "field": ["q", "r", "s"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                    "most_common_individuals": ["h", "m", "i", "j", "k", "o", "p", "q", "r", "s"],
                },
                {
                    "field": ["k", "h", "m"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                    "most_common_individuals": ["h", "m", "i", "j", "k", "o", "p", "q", "r", "s"],
                },
                {
                    "field": ["h", "i", "j"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                    "most_common_individuals": ["h", "m", "i", "j", "k", "o", "p", "q", "r", "s"],
                },
                {
                    "field": ["q", "r", "s"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                    "most_common_individuals": ["h", "m", "i", "j", "k", "o", "p", "q", "r", "s"],
                },
                {
                    "field": ["k", "h", "m"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                    "most_common_individuals": ["h", "m", "i", "j", "k", "o", "p", "q", "r", "s"],
                },
                {
                    "field": ["m", "o", "p"],
                    "most_common_lists": [["h", "i", "j"], ["k", "h", "m"], ["m", "o", "p"], ["q", "r", "s"]],
                    "most_common_individuals": ["h", "m", "i", "j", "k", "o", "p", "q", "r", "s"],
                },
            ],
        }
        self.assertDictEqual(
            output_multi_stream,
            expected_output4,
            "expected to see: \n"
            + json.dumps(expected_output4)
            + "\n but instead, received: \n"
            + json.dumps(output_multi_stream),
        )

        with self.assertRaises(ValueError):
            output_multi_stream = ExtractFieldValues(
                stream_name="train",
                field="animal",
                to_field="most_common_individuals",
                overall_top_frequency_percent=90,
                process_every_value=True,
            ).process(input_multi_stream1)

        with self.assertRaises(AssertionError):
            output_multi_stream = ExtractFieldValues(
                stream_name="train",
                field="animal",
                to_field="most_common_individuals",
                overall_top_frequency_percent=90,
                min_frequency_percent=25,
            ).process(input_multi_stream1)
        with self.assertRaises(AssertionError):
            output_multi_stream = ExtractFieldValues(
                stream_name="train",
                field="animal",
                to_field="most_common_individuals",
                overall_top_frequency_percent=120,
            ).process(input_multi_stream1)
        with self.assertRaises(AssertionError):
            output_multi_stream = ExtractFieldValues(
                stream_name="train",
                field="animal",
                to_field="most_common_individuals",
                min_frequency_percent=-2,
            ).process(input_multi_stream1)

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

    def test_deterministic_balancer_empty_stream(self):
        inputs = []

        targets = []

        test_operator(
            operator=DeterministicBalancer(fields=["a", "b"]),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

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

    def test_augment_whitespace_model_input(self):
        source = "The dog ate my cat"
        inputs = [{"source": source}]

        operator = AugmentWhitespace(augment_model_input=True)
        outputs = apply_operator(operator, inputs)
        assert outputs[0]["source"] != source, f"Source of f{outputs} is equal to f{source} and was not augmented"
        normalized_output_source = outputs[0]["source"].split()
        normalized_input_source = source.split()
        assert (
            normalized_output_source == normalized_input_source
        ), f"{normalized_output_source} is not equal to f{normalized_input_source}"

    def test_augment_whitespace_task_input_with_error(self):
        text = "The dog ate my cat"
        inputs = [{"inputs": {"text": text}}]
        operator = AugmentWhitespace(augment_task_input=True)
        operator.set_task_input_fields(["sentence"])
        with self.assertRaises(ValueError):
            outputs = apply_operator(operator, inputs)

    def test_augment_whitespace_task_input(self):
        text = "The dog ate my cat"
        inputs = [{"inputs": {"text": text}}]
        operator = AugmentWhitespace(augment_task_input=True)
        operator.set_task_input_fields(["text"])
        outputs = apply_operator(operator, inputs)
        normalized_output_source = outputs[0]["inputs"]["text"].split()
        normalized_input_source = text.split()
        assert (
            normalized_output_source == normalized_input_source
        ), f"{normalized_output_source} is not equal to f{normalized_input_source}"

    def test_augment_whitespace_with_none_text_error(self):
        text = None
        inputs = [{"inputs": {"text": text}}]
        operator = AugmentWhitespace(augment_task_input=True)
        operator.set_task_input_fields(["text"])
        exception_text = "Error processing instance '0' from stream 'test' in AugmentWhitespace due to: Error augmenting value 'None' from 'inputs/text' in instance: {'inputs': {'text': None}}"
        test_operator_exception(
            operator,
            inputs,
            tester=self,
            exception_text=exception_text,
        )

    def test_test_operator_without_tester_param(self):
        text = None
        inputs = [{"inputs": {"text": text}}]
        operator = AugmentWhitespace(augment_task_input=True)
        operator.set_task_input_fields(["text"])
        exception_text = "Error processing instance '0' from stream 'test' in AugmentWhitespace due to: Error augmenting value 'None' from 'inputs/text' in instance: {'inputs': {'text': None}}"

        test_operator_exception(
            operator,
            inputs,
            exception_text=exception_text,
        )

    def test_test_operator_unexpected_pass(self):
        text = "Should be ok"
        inputs = [{"inputs": {"text": text}}]
        operator = AugmentWhitespace(augment_task_input=True)
        operator.set_task_input_fields(["text"])
        exception_text = "Error processing instance '0' from stream 'test' in AugmentWhitespace due to: Error augmenting value 'None' from 'inputs/text' in instance: {'inputs': {'text': None}}"

        try:
            test_operator_exception(
                operator,
                inputs,
                exception_text=exception_text,
            )
        except Exception as e:
            self.assertEqual(
                str(e),
                "Did not receive expected exception Error processing instance '0' from stream 'test' in AugmentWhitespace due to: Error augmenting value 'None' from 'inputs/text' in instance: {'inputs': {'text': None}}",
            )
