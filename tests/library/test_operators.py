import json
from collections import Counter
from typing import Any

from unitxt.formats import SystemFormat
from unitxt.metrics import MetricsList
from unitxt.operators import (
    AddConstant,
    Apply,
    ApplyMetric,
    ApplyOperatorsField,
    ApplyStreamOperatorsField,
    Balance,
    CastFields,
    CollateInstances,
    Copy,
    DeterministicBalancer,
    DivideAllFieldsBy,
    DuplicateInstances,
    EncodeLabels,
    ExecuteExpression,
    ExtractFieldValues,
    ExtractMostCommonFieldValues,
    FeatureGroupedShuffle,
    FieldOperator,
    FilterByCondition,
    FilterByConditionBasedOnFields,
    FilterByExpression,
    FlattenInstances,
    FromIterables,
    IndexOf,
    Intersect,
    IterableSource,
    JoinStr,
    LengthBalancer,
    ListFieldValues,
    MapInstanceValues,
    MergeStreams,
    MinimumOneExamplePerLabelRefiner,
    Perturb,
    RemoveFields,
    RemoveValues,
    Rename,
    SelectFields,
    Set,
    Shuffle,
    ShuffleFieldValues,
    SplitByValue,
    StreamRefiner,
    TakeByField,
    Unique,
    ZipFieldValues,
)
from unitxt.stream import MultiStream
from unitxt.stream_operators import DeleteSplits, JoinStreams
from unitxt.templates import InputOutputTemplate, MultiReferenceTemplate
from unitxt.test_utils.operators import (
    apply_operator,
    check_operator,
    check_operator_exception,
)

from tests.utils import UnitxtTestCase


class TestOperators(UnitxtTestCase):
    def compare_streams(self, all, expected_all):
        self.assertEqual(len(all), len(expected_all))
        for input_dict, output_dict in zip(all, expected_all):
            self.assertDictEqual(input_dict, output_dict)

    def test_map_instance_values(self):
        mappers = {"a": {"1": "hi", "2": "bye"}}

        inputs = [
            {"a": "1", "b": "2"},
            {"a": "2", "b": "3"},
        ]

        targets = [
            {"a": "hi", "b": "2"},
            {"a": "bye", "b": "3"},
        ]

        # simple value substitute
        check_operator(
            operator=MapInstanceValues(mappers=mappers),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

        # process_every_value=True would not accept non-list inputs
        check_operator_exception(
            operator=MapInstanceValues(mappers=mappers, process_every_value=True),
            inputs=inputs,
            exception_texts=[
                "Error processing instance '0' from stream 'test' in MapInstanceValues due to the exception above.",
                "'process_every_field' == True is allowed only for fields whose values are lists, but value of field 'a' is '1'",
            ],
            tester=self,
        )

        # strict is True by default, input value "3" in field "a" is missing from the mapper of "a"
        check_operator_exception(
            operator=MapInstanceValues(mappers=mappers),
            inputs=[{"a": "3", "b": "4"}],
            exception_texts=[
                "Error processing instance '0' from stream 'test' in MapInstanceValues due to the exception above.",
                "\"value '3', the string representation of the value in field 'a', is not found in mapper '{'1': 'hi', '2': 'bye'}'\"",
            ],
            tester=self,
        )

        inputs_process_every_value = [
            {"a": [1, 2, 3, 4], "b": 2},
            {"a": [2], "b": 3},
        ]

        targets_process_every_value = [
            {"a": ["hi", "bye", 3, 4], "b": 2},
            {"a": ["bye"], "b": 3},
        ]

        # simple mapping of individual elements in the list. strict is False here, to ignore absence of "3" from the mapper of "a"
        check_operator(
            operator=MapInstanceValues(
                mappers=mappers, process_every_value=True, strict=False
            ),
            inputs=inputs_process_every_value,
            targets=targets_process_every_value,
            tester=self,
        )

        # simple mapping of individual elements in the list. with strict=True, the absence of "3" from the mapper of "a" is not overlooked
        check_operator_exception(
            operator=MapInstanceValues(mappers=mappers, process_every_value=True),
            inputs=[{"a": [1, 2, 3, 4], "b": 2}],
            exception_texts=[
                "Error processing instance '0' from stream 'test' in MapInstanceValues due to the exception above.",
                "\"value '3', the string representation of the value in field 'a', is not found in mapper '{'1': 'hi', '2': 'bye'}'\"",
            ],
            tester=self,
        )
        # Test mapping of lists to lists
        inputs_not_process_every_value = [
            {"a": [1, 2, 3, 4], "b": 2},
            {"a": [], "b": 3},
        ]

        targets_not_process_every_value = [
            {"a": ["All"], "b": 2},
            {"a": ["None"], "b": 3},
        ]

        list_mappers = {"a": {str([1, 2, 3, 4]): ["All"], "[]": ["None"]}}
        check_operator(
            operator=MapInstanceValues(
                mappers=list_mappers, process_every_value=False, strict=False
            ),
            inputs=inputs_not_process_every_value,
            targets=targets_not_process_every_value,
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

        check_operator(
            operator=MapInstanceValues(mappers={"a": {"1": "hi", "2": "bye"}}),
            inputs=inputs,
            targets=targets,
        )

    def test_map_instance_values_with_non_str_returned_value(self):
        inputs = [
            {"a": 1, "b": 2},
            {"a": 2, "b": 3},
        ]

        targets = [
            {"a": {"word": 4, "words": [5, 6, 7]}, "b": 2},
            {"a": ["bye", "ciao", [0, 1, 2]], "b": 3},
        ]

        operator = MapInstanceValues(
            mappers={
                "a": {
                    "1": {"word": 4, "words": [5, 6, 7]},
                    "2": ["bye", "ciao", [0, 1, 2]],
                }
            }
        )
        outputs = check_operator(
            operator=operator,
            inputs=inputs,
            targets=targets,
        )

        # tweak the returned values in outputs:
        outputs[0]["a"]["words"][0] = 5000
        outputs[1]["a"][2] = []
        # and run operator again, to verify that its mapper have not changed
        check_operator(
            operator=operator,
            inputs=inputs,
            targets=targets,
        )

    def test_from_iterables_and_iterable_source(self):
        input_ms = {
            "train": [{"a": "1"}, {"b": "2"}, {"a": "3"}],
            "test": [{"a": "4"}, {"c": "5"}, {"a": "6"}],
        }

        operator = FromIterables()
        output_ms = operator.process(input_ms)
        self.assertSetEqual(set(input_ms.keys()), set(output_ms.keys()))
        for stream_name in input_ms.keys():
            self.assertListEqual(
                list(input_ms[stream_name]), list(output_ms[stream_name])
            )

        # IterableSource is a callable
        operator = IterableSource(iterables=input_ms)
        output_ms = operator()
        self.assertSetEqual(set(input_ms.keys()), set(output_ms.keys()))
        for stream_name in input_ms.keys():
            self.assertListEqual(
                list(input_ms[stream_name]), list(output_ms[stream_name])
            )

    def test_list_field_values(self):
        inputs = [
            {"a": 1, "b": 2, "c": 3},
            {"a": 2, "b": 3, "c": 5},
        ]

        targets = [
            {"a": 1, "b": 2, "ab": [1, 2], "c": 3},
            {"a": 2, "b": 3, "ab": [2, 3], "c": 5},
        ]

        check_operator(
            operator=ListFieldValues(fields=["a", "b"], to_field="ab"),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_field_operator(self):
        class ExpandJustForCoverage(FieldOperator):
            def process_value(self, value: Any) -> Any:
                super().process_value(value)
                pass

        ExpandJustForCoverage(field_to_field={"from": "to"}).process_value(2)

        class ExpandJustForCoverage2(FieldOperator):
            def process_value(self, value: Any) -> Any:
                return str(value).upper()

        inputs = [
            {"a": "imagine", "b": ["there's", "no", "heaven"]},
            {"a": "imagine", "b": ["all", "the", "people"]},
        ]

        targets = [
            {
                "a": "imagine",
                "b": ["there's", "no", "heaven"],
                "B": ["THERE'S", "NO", "HEAVEN"],
            },
            {
                "a": "imagine",
                "b": ["all", "the", "people"],
                "B": ["ALL", "THE", "PEOPLE"],
            },
        ]

        check_operator(
            operator=ExpandJustForCoverage2(
                field_to_field=[["b", "B"]], process_every_value=True
            ),
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

        check_operator(
            operator=FlattenInstances(sep="..."),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_filter_by_values_with_required_values(self):
        inputs = [
            {"a": 1, "b": {"c": 2}},
            {"a": 2, "b": {"c": 3}},
            {"a": 1, "b": {"c": 3}},
        ]

        targets = [
            {"a": 1, "b": {"c": 3}},
        ]

        check_operator(
            operator=FilterByCondition(values={"a": 1, "b/c": 3}, condition="eq"),
            inputs=inputs,
            targets=targets,
            tester=self,
        )
        check_operator(
            operator=FilterByExpression(expression="a == 1 and b['c'] == 3"),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

        exception_text = (
            "Required filter field ('d') in FilterByCondition is not found in instance."
        )
        check_operator_exception(
            operator=FilterByCondition(values={"d": "5"}, condition="eq"),
            inputs=inputs,
            exception_texts=[exception_text],
            tester=self,
        )
        check_operator_exception(
            operator=FilterByExpression(expression="d['e'] == 5"),
            inputs=inputs,
            exception_texts=["name 'd' is not defined"],
            tester=self,
        )

    def test_filter_by_condition_ne(self):
        inputs = [
            {"a": [0, 10], "b": 2},
            {"a": [2, 20], "b": 3},
            {"a": [1, 100], "b": 3},
        ]

        targets = [
            {"a": [2, 20], "b": 3},
        ]

        check_operator(
            operator=FilterByCondition(values={"a/0": 1, "b": 2}, condition="ne"),
            inputs=inputs,
            targets=targets,
            tester=self,
        )
        check_operator(
            operator=FilterByExpression(expression="a[0] != 1 and b != 2"),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_filter_by_condition_gt(self):
        inputs = [{"a": 0, "b": 2}, {"a": 2, "b": 3}, {"a": 1, "b": 3}]

        targets = [
            {"a": 2, "b": 3},
        ]

        check_operator(
            operator=FilterByCondition(values={"a": 1}, condition="gt"),
            inputs=inputs,
            targets=targets,
            tester=self,
        )
        check_operator(
            operator=FilterByExpression(expression="a>1"),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_filter_by_condition_bad_condition(self):
        with self.assertRaises(ValueError):
            FilterByCondition(values={"a": 1}, condition="gte")

    def test_filter_by_condition_not_in(self):
        inputs = [
            {"a": 1, "b": 2},
            {"a": 2, "b": 3},
            {"a": 3, "b": 4},
        ]

        targets = [
            {"a": 1, "b": 2},
        ]

        check_operator(
            operator=FilterByCondition(values={"b": [3, 4]}, condition="not in"),
            inputs=inputs,
            targets=targets,
            tester=self,
        )
        check_operator(
            operator=FilterByExpression(expression="b not in [3, 4]"),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_filter_by_condition_not_in_multiple(self):
        inputs = [
            {"a": 1, "b": 2},
            {"a": 2, "b": 3},
            {"a": 3, "b": 4},
        ]

        targets = []

        check_operator(
            operator=FilterByCondition(
                values={"b": [3, 4], "a": [1]},
                condition="not in",
                error_on_filtered_all=False,
            ),
            inputs=inputs,
            targets=targets,
            tester=self,
        )
        check_operator(
            operator=FilterByExpression(
                expression="b not in [3, 4] and a not in [1]",
                error_on_filtered_all=False,
            ),
            inputs=inputs,
            targets=targets,
            tester=self,
        )
        check_operator_exception(
            operator=FilterByExpression(
                expression="b not in [3, 4] and a not in [1]",
                error_on_filtered_all=True,
            ),
            inputs=inputs,
            exception_texts=[
                "FilterByExpression filtered out every instance in stream 'test'. If this is intended set error_on_filtered_all=False"
            ],
            tester=self,
        )

    def test_filter_by_condition_in(self):
        inputs = [
            {"a": 1, "b": 2},
            {"a": 2, "b": 3},
            {"a": 3, "b": 4},
        ]

        targets = [
            {"a": 2, "b": 3},
            {"a": 3, "b": 4},
        ]

        check_operator(
            operator=FilterByCondition(values={"b": [3, 4]}, condition="in"),
            inputs=inputs,
            targets=targets,
            tester=self,
        )
        check_operator(
            operator=FilterByExpression(expression="b in [3, 4]"),
            inputs=inputs,
            targets=targets,
            tester=self,
        )
        with self.assertRaises(ValueError) as cm:
            check_operator(
                operator=FilterByCondition(values={"b": "5"}, condition="in"),
                inputs=inputs,
                targets=targets,
                tester=self,
            )
        self.assertEqual(
            str(cm.exception),
            "The filter for key ('b') in FilterByCondition with condition 'in' must be list but is not : '5'",
        )

        with self.assertRaises(ValueError) as cm:
            check_operator(
                operator=FilterByCondition(values={"c": ["5"]}, condition="in"),
                inputs=inputs,
                targets=targets,
                tester=self,
            )
        self.assertEqual(
            str(cm.exception),
            "Required filter field ('c') in FilterByCondition is not found in instance.",
        )
        with self.assertRaises(Exception) as ne:
            check_operator(
                operator=FilterByExpression(expression="c in ['5']"),
                inputs=inputs,
                targets=targets,
                tester=self,
            )
        self.assertEqual("name 'c' is not defined", str(ne.exception))

    def test_filter_by_condition_error_when_the_entire_stream_is_filtered(self):
        inputs = [{"a": 1, "b": 2}, {"a": 2, "b": 3}, {"a": 1, "b": 3}]
        with self.assertRaises(RuntimeError) as e:
            check_operator(
                operator=FilterByCondition(
                    values={"b": ["weird_value"]}, condition="in"
                ),
                inputs=inputs,
                targets=[],
                tester=self,
            )
        self.assertEqual(
            str(e.exception),
            "FilterByCondition filtered out every instance in stream 'test'. If this is intended set error_on_filtered_all=False",
        )

    def test_execute_expression(self):
        inputs = [{"a": 2, "b": 3}]
        operator = ExecuteExpression(to_field="c", expression="a+b")
        targets = [{"a": 2, "b": 3, "c": 5}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)
        inputs = [{"a": "hello", "b": "world"}]
        operator = ExecuteExpression(expression="a+' '+b", to_field="c")
        targets = [{"a": "hello", "b": "world", "c": "hello world"}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)
        operator = ExecuteExpression(expression="f'{a} {b}'", to_field="c")
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)
        check_operator_exception(
            operator=operator,
            inputs=[{"x": 2, "y": 3}],
            exception_texts=[
                "Error processing instance '0' from stream 'test' in ExecuteExpression due to the exception above.",
                "name 'a' is not defined",
            ],
            tester=self,
        )

        inputs = [{"json_string": '{"A":"a_value", "B":"b_value"}'}]
        operator = ExecuteExpression(
            expression='json.loads(json_string)["A"]',
            imports_list=["json"],
            to_field="c",
        )
        self.assertEqual("a_value", operator.process(inputs[0])["c"])

        pattern = "[0-9]+"
        string = "Account Number - 12345, Amount - 586.32"
        repl = "NN"
        inputs = [{"pattern": pattern, "string": string, "repl": repl}]
        operator = ExecuteExpression(
            expression="re.sub(pattern, repl, string)",
            imports_list=["re"],
            to_field="c",
        )
        self.assertEqual(
            "Account Number - NN, Amount - NN.NN", operator.process(inputs[0])["c"]
        )

    def test_intersect(self):
        
        def __test_intersect(inputs, targets, fields_to_intersect, allowed_field_values):
            return check_operator(
                operator=Intersect(fields_to_intersect, allowed_field_values),
                inputs=inputs,
                targets=targets,
                )
        
        ## basic test
        __test_intersect(
            inputs=[{"label": [1,2]}], 
            targets=[{"label": [1]}],
            fields_to_intersect=["label"],
            allowed_field_values=[1]
                        )

        # multiple fields of the same name
        __test_intersect(
            inputs = [
                {"label": ["a", "b"]},
                {"label": ["a", "c", "d"]},
                {"name": ["a", "b", "f"]},
            ],
            targets = [
                {"label": ["b"]},
                {"label": []},
                {"name": ["b", "f"]},
            ],
            fields_to_intersect=["label",'name'],
            allowed_field_values=["b", "f"]
        )
        
        __test_intersect(
            inputs = [
                {"label": ["a", "b"]},
                {"label": ["a", "c", "d"]},
                {"label": ["a", "b", "f"]},
            ],
            targets = [
                {"label": ["b"]},
                {"label": []},
                {"label": ["b", "f"]},
            ],
            fields_to_intersect=["label"], 
            allowed_field_values=["b", "f"]
        )
        
        
        with self.assertRaises(ValueError) as cm:
            __test_intersect(
                inputs = [
                    {"label": ["a", "b"]},
                ],
                targets = [
                {"label": ["b"]},
                ],
                fields_to_intersect=["label"], 
                allowed_field_values=3
                )
        self.assertEqual(str(cm.exception), "The allowed_field_values is not a list but '<class 'int'>'")

        
        inputs = [
            {"label": "b"},
        ]
        exception_texts = [
            "Error processing instance '0' from stream 'test' in Intersect due to the exception above.",
            "Failed to process field 'label' from instance due to the exception above.",
            "The value in field is not a list but 'b'",
        ]
        check_operator_exception(
            operator=Intersect(field=["label"], allowed_field_values=["c"]),
            inputs=inputs,
            exception_texts=exception_texts,
            tester=self,
        )

    def test_remove_none(self):
        inputs = [
            {"references": [["none"], ["none"]]},
            {"references": [["news", "games"], ["none"]]},
        ]

        targets = [
            {"references": [[], ["none"]]},
            {"references": [["news", "games"], ["none"]]},
        ]

        check_operator(
            operator=RemoveValues(
                field="references/0",
                unallowed_values=["none"],
                process_every_value=False,
            ),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

        check_operator(
            operator=RemoveValues(
                field="references/1",
                unallowed_values=["none"],
                process_every_value=False,
            ),
            inputs=inputs,
            targets=[
                {"references": [["none"], []]},
                {"references": [["news", "games"], []]},
            ],
            tester=self,
        )

        check_operator(
            operator=RemoveValues(
                field="references",
                unallowed_values=["none"],
                process_every_value=True,
            ),
            inputs=inputs,
            targets=[
                {"references": [[], []]},
                {"references": [["news", "games"], []]},
            ],
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

        check_operator(
            operator=RemoveValues(field="label", unallowed_values=["b", "f"]),
            inputs=inputs,
            targets=targets,
            tester=self,
        )
        with self.assertRaises(ValueError) as cm:
            check_operator(
                operator=RemoveValues(field="label", unallowed_values=3),
                inputs=inputs,
                targets=targets,
                tester=self,
            )
        self.assertEqual(
            str(cm.exception), "The unallowed_values is not a list but '3'"
        )

        inputs = [
            {"label": "b"},
        ]
        exception_texts = [
            "Error processing instance '0' from stream 'test' in RemoveValues due to the exception above.",
            "Failed to process field 'label' from instance due to the exception above.",
            "The value in field is not a list but 'b'",
        ]
        check_operator_exception(
            operator=RemoveValues(field="label", unallowed_values=["c"]),
            inputs=inputs,
            exception_texts=exception_texts,
            tester=self,
        )

        exception_texts = [
            "Error processing instance '0' from stream 'test' in RemoveValues due to the exception above.",
            "Failed to get 'label2' from instance due to the exception above.",
            """query "label2" did not match any item in dict:
label (str):
    b
""",
        ]
        check_operator_exception(
            operator=RemoveValues(field="label2", unallowed_values=["c"]),
            inputs=inputs,
            exception_texts=exception_texts,
            tester=self,
        )

    def test_apply_value_operators_field(self):
        inputs = [
            {
                "prediction": 111,
                "references": [],
                "b": 2,
                "c": ["processors.to_string", "processors.first_character"],
            },
            {
                "prediction": 222,
                "references": [],
                "b": 3,
                "c": ["processors.to_string", "processors.first_character"],
            },
        ]

        targets = [
            {
                "prediction": "1",
                "references": [],
                "b": 2,
                "c": ["processors.to_string", "processors.first_character"],
            },
            {
                "prediction": "2",
                "references": [],
                "b": 3,
                "c": ["processors.to_string", "processors.first_character"],
            },
        ]

        # the expression of the operator names changes, so we check correctness of data fields only:
        operator = ApplyOperatorsField(operators_field="c")
        outputs = list(operator(MultiStream.from_iterables({"tmp": inputs}))["tmp"])
        self.assertEqual(len(outputs), len(targets))
        for output, target in zip(outputs, targets):
            self.assertEqual(output["prediction"], target["prediction"])
            self.assertEqual(output["references"], target["references"])
            self.assertEqual(output["b"], target["b"])

        # check the case no operators are specified in field operators_field. default_operators is none by default
        check_operator_exception(
            operator=ApplyOperatorsField(operators_field="d"),
            inputs=inputs,
            exception_texts=[
                "Error processing instance '0' from stream 'test' in ApplyOperatorsField due to the exception above.",
                "No operators found in field 'd', and no default operators provided.",
            ],
            tester=self,
        )
        # check default operators:
        inputs = [
            {"prediction": 111, "references": [222, 333]},
            {"prediction": 222, "references": [999]},
        ]
        operator = ApplyOperatorsField(
            operators_field="d", default_operators="processors.to_string"
        )
        targets = [
            {"prediction": "111", "references": ["222", "333"]},
            {"prediction": "222", "references": ["999"]},
        ]
        outputs = list(operator(MultiStream.from_iterables({"tmp": inputs}))["tmp"])
        self.assertEqual(len(outputs), len(targets))
        for output, target in zip(outputs, targets):
            self.assertEqual(output["prediction"], target["prediction"])
            self.assertEqual(output["references"], target["references"])

    def test_set(self):
        inputs = [
            {"a": 1, "b": 2},
            {"a": 2, "b": 3},
        ]

        targets = [
            {"a": 1, "b": 2, "c": 3},
            {"a": 2, "b": 3, "c": 3},
        ]

        check_operator(
            operator=Set(fields={"c": 3}),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_set_with_query(self):
        inputs = [
            {"a": {"a": 1, "b": 2}, "b": 2},
            {"a": {"a": 2, "b": 3}, "b": 3},
        ]

        targets = [
            {"a": {"a": 1, "b": 2, "c": 5}, "b": 2},
            {"a": {"a": 2, "b": 3, "c": 5}, "b": 3},
        ]

        check_operator(
            operator=Set(fields={"a/c": 5}),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_set_with_deep_copy(self):
        inputs = [
            {"a": 1, "b": 2},
            {"a": 2, "b": 3},
        ]

        alist = [4]

        targets = [
            {"a": 1, "b": 2, "c": [4]},
            {"a": 2, "b": 3, "c": [4]},
        ]

        outputs = check_operator(
            operator=Set(fields={"c": alist}, use_deepcopy=True),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

        alist.append(5)

        self.assertDictEqual(outputs[0], targets[0])

        targets = [
            {"a": 1, "b": 2, "c": {"d": [4, 5]}},
            {"a": 2, "b": 3, "c": {"d": [4, 5]}},
        ]

        outputs = check_operator(
            operator=Set(fields={"c/d": alist}, use_deepcopy=True),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

        alist.append(6)

        self.assertDictEqual(outputs[0], targets[0])

    def test_remove_fields(self):
        inputs = [
            {"a": 1, "b": 2},
            {"a": 2, "b": 3},
        ]

        targets = [
            {"a": 1},
            {"a": 2},
        ]

        check_operator(
            operator=RemoveFields(fields=["b"]),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_unique_on_single_field(self):
        inputs = [
            {"a": [1, 5], "b": 2},
            {"a": [2, 5], "b": 3},
            {"a": [2, 5], "b": 4},
        ]

        targets = {((1, 5),), ((2, 5),)}

        outputs = apply_operator(
            operator=Unique(fields=["a"]),
            inputs=inputs,
        )

        self.assertSetEqual(set(outputs), targets)

    def test_apply_stream_operators_field(self):
        inputs = [
            {
                "prediction": "IMAGINE  ",
                "references": ["IMAGINE  ", "IMAGINE  ", "IMAGINE  "],
                "operator": "processors.lower_case",
            },
            {"prediction": "ALL  ", "references": ["ALL  ", "ALL  "]},
            {"prediction": "  The  ", "references": ["  The  ", "  The  "]},
            {"prediction": " peOple  ", "references": [" peOple  ", " peOple  "]},
        ]
        operator = ApplyStreamOperatorsField(field="operator", reversed=True)
        outputs = list(
            operator.process(MultiStream.from_iterables({"train": inputs})["train"])
        )
        self.assertListEqual(
            [
                {
                    "operator": "processors.lower_case",
                    "prediction": "imagine  ",
                    "references": ["imagine  ", "imagine  ", "imagine  "],
                },
                {"prediction": "all  ", "references": ["all  ", "all  "]},
                {"prediction": "  the  ", "references": ["  the  ", "  the  "]},
                {"prediction": " people  ", "references": [" people  ", " people  "]},
            ],
            outputs,
        )

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

        outputs = apply_operator(
            operator=SplitByValue(fields="a"), inputs=inputs, return_multi_stream=True
        )

        self.assertSetEqual(set(outputs.keys()), {"test_1", "test_2"})

        outputs_1 = list(outputs["test_1"])
        self.assertEqual(len(outputs_1), 1)

        outputs_2 = list(outputs["test_2"])
        self.assertEqual(len(outputs_2), 2)

        for input_dict, output_dict in zip(inputs, outputs_1):
            self.assertDictEqual(input_dict, output_dict)

        for input_dict, output_dict in zip(inputs[1:], outputs_2):
            self.assertDictEqual(input_dict, output_dict)

    def test_merge(self):
        # Test with default params
        input_multi_stream = MultiStream(
            {
                "test": [{"field": "test1"}],
                "validation": [{"field": "validation1"}],
                "train": [{"field": "train1"}],
            }
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
            {
                "test": [{"field": "test1"}],
                "validation": [{"field": "validation1"}],
                "train": [{"field": "train1"}],
            }
        )
        output_multi_stream = MergeStreams(
            streams_to_merge=["test", "train"],
            new_stream_name="merged",
            add_origin_stream_name=False,
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
                    {"animal": "cat"},
                    {"animal": "dog"},
                    {"animal": "cat"},
                    {"animal": "sheep"},
                    {"animal": "fish"},
                    {"animal": "shark"},
                ],
            }
        )
        output_multi_stream1 = ExtractFieldValues(
            stream_name="train",
            field="animal",
            to_field="all_animals1",
        ).process(input_multi_stream1)
        output_for_comparison1 = {}
        for k, v in output_multi_stream1.items():
            output_for_comparison1[k] = list(v)
        expected_output1 = {
            "test": [
                {
                    "animal": "shark",
                    "all_animals1": ["fish", "dog", "cat", "sheep", "shark"],
                }
            ],
            "validation": [
                {
                    "animal": "cat",
                    "all_animals1": ["fish", "dog", "cat", "sheep", "shark"],
                }
            ],
            "train": [
                {
                    "animal": "fish",
                    "all_animals1": ["fish", "dog", "cat", "sheep", "shark"],
                },
                {
                    "animal": "dog",
                    "all_animals1": ["fish", "dog", "cat", "sheep", "shark"],
                },
                {
                    "animal": "cat",
                    "all_animals1": ["fish", "dog", "cat", "sheep", "shark"],
                },
                {
                    "animal": "dog",
                    "all_animals1": ["fish", "dog", "cat", "sheep", "shark"],
                },
                {
                    "animal": "cat",
                    "all_animals1": ["fish", "dog", "cat", "sheep", "shark"],
                },
                {
                    "animal": "sheep",
                    "all_animals1": ["fish", "dog", "cat", "sheep", "shark"],
                },
                {
                    "animal": "fish",
                    "all_animals1": ["fish", "dog", "cat", "sheep", "shark"],
                },
                {
                    "animal": "shark",
                    "all_animals1": ["fish", "dog", "cat", "sheep", "shark"],
                },
            ],
        }
        self.assertDictEqual(
            output_for_comparison1,
            expected_output1,
            "expected to see: \n"
            + json.dumps(expected_output1)
            + "\n but instead, received: \n"
            + json.dumps(output_for_comparison1),
        )

    def test_extract_most_common_values(self):
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
        output_multi_stream1 = ExtractMostCommonFieldValues(
            stream_name="train",
            field="animal",
            to_field="most_common_animals1",
            overall_top_frequency_percent=80,
        ).process(input_multi_stream1)
        output_for_comparison1 = {}
        for k, v in output_multi_stream1.items():
            output_for_comparison1[k] = list(v)
        expected_output1 = {
            "test": [
                {"animal": "shark", "most_common_animals1": ["dog", "cat", "fish"]}
            ],
            "validation": [
                {"animal": "cat", "most_common_animals1": ["dog", "cat", "fish"]}
            ],
            "train": [
                {"animal": "fish", "most_common_animals1": ["dog", "cat", "fish"]},
                {"animal": "dog", "most_common_animals1": ["dog", "cat", "fish"]},
                {"animal": "dog", "most_common_animals1": ["dog", "cat", "fish"]},
                {"animal": "cat", "most_common_animals1": ["dog", "cat", "fish"]},
                {"animal": "dog", "most_common_animals1": ["dog", "cat", "fish"]},
                {"animal": "cat", "most_common_animals1": ["dog", "cat", "fish"]},
                {"animal": "sheep", "most_common_animals1": ["dog", "cat", "fish"]},
                {"animal": "cat", "most_common_animals1": ["dog", "cat", "fish"]},
                {"animal": "fish", "most_common_animals1": ["dog", "cat", "fish"]},
                {"animal": "shark", "most_common_animals1": ["dog", "cat", "fish"]},
            ],
        }
        self.assertDictEqual(
            output_for_comparison1,
            expected_output1,
            "expected to see: \n"
            + json.dumps(expected_output1)
            + "\n but instead, received: \n"
            + json.dumps(output_for_comparison1),
        )
        # with minimum frequency limit
        output_multi_stream2 = ExtractMostCommonFieldValues(
            stream_name="train",
            field="animal",
            to_field="most_common_animals2",
            min_frequency_percent=25,
        ).process(input_multi_stream1)
        output_for_comparison2 = {}
        for k, v in output_multi_stream2.items():
            output_for_comparison2[k] = list(v)
        expected_output2 = {
            "test": [
                {
                    "animal": "shark",
                    "most_common_animals1": ["dog", "cat", "fish"],
                    "most_common_animals2": ["dog", "cat"],
                }
            ],
            "validation": [
                {
                    "animal": "cat",
                    "most_common_animals1": ["dog", "cat", "fish"],
                    "most_common_animals2": ["dog", "cat"],
                }
            ],
            "train": [
                {
                    "animal": "fish",
                    "most_common_animals1": ["dog", "cat", "fish"],
                    "most_common_animals2": ["dog", "cat"],
                },
                {
                    "animal": "dog",
                    "most_common_animals1": ["dog", "cat", "fish"],
                    "most_common_animals2": ["dog", "cat"],
                },
                {
                    "animal": "dog",
                    "most_common_animals1": ["dog", "cat", "fish"],
                    "most_common_animals2": ["dog", "cat"],
                },
                {
                    "animal": "cat",
                    "most_common_animals1": ["dog", "cat", "fish"],
                    "most_common_animals2": ["dog", "cat"],
                },
                {
                    "animal": "dog",
                    "most_common_animals1": ["dog", "cat", "fish"],
                    "most_common_animals2": ["dog", "cat"],
                },
                {
                    "animal": "cat",
                    "most_common_animals1": ["dog", "cat", "fish"],
                    "most_common_animals2": ["dog", "cat"],
                },
                {
                    "animal": "sheep",
                    "most_common_animals1": ["dog", "cat", "fish"],
                    "most_common_animals2": ["dog", "cat"],
                },
                {
                    "animal": "cat",
                    "most_common_animals1": ["dog", "cat", "fish"],
                    "most_common_animals2": ["dog", "cat"],
                },
                {
                    "animal": "fish",
                    "most_common_animals1": ["dog", "cat", "fish"],
                    "most_common_animals2": ["dog", "cat"],
                },
                {
                    "animal": "shark",
                    "most_common_animals1": ["dog", "cat", "fish"],
                    "most_common_animals2": ["dog", "cat"],
                },
            ],
        }
        self.assertDictEqual(
            output_for_comparison2,
            expected_output2,
            "expected to see: \n"
            + json.dumps(expected_output2)
            + "\n but instead, received: \n"
            + json.dumps(output_for_comparison2),
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
        output_multi_stream3 = ExtractMostCommonFieldValues(
            stream_name="train",
            field="field",
            to_field="most_common_lists3",
            overall_top_frequency_percent=90,
            process_every_value=False,
        ).process(input_multi_stream2)
        output_for_comparison3 = {}
        for k, v in output_multi_stream3.items():
            output_for_comparison3[k] = list(v)
        expected_output3 = {
            "test": [
                {
                    "field": ["a", "b", "c"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                }
            ],
            "validation": [
                {
                    "field": ["d", "e", "f"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                }
            ],
            "train": [
                {
                    "field": ["t", "u", "v"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                },
                {
                    "field": ["h", "i", "j"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                },
                {
                    "field": ["k", "h", "m"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                },
                {
                    "field": ["m", "o", "p"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                },
                {
                    "field": ["m", "o", "p"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                },
                {
                    "field": ["h", "i", "j"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                },
                {
                    "field": ["q", "r", "s"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                },
                {
                    "field": ["k", "h", "m"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                },
                {
                    "field": ["h", "i", "j"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                },
                {
                    "field": ["q", "r", "s"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                },
                {
                    "field": ["k", "h", "m"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                },
                {
                    "field": ["m", "o", "p"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                },
            ],
        }

        self.assertDictEqual(
            output_for_comparison3,
            expected_output3,
            "expected to see: \n"
            + json.dumps(expected_output3)
            + "\n but instead, received: \n"
            + json.dumps(output_for_comparison3),
        )

        # finally, with lists and with process_every_value=True
        output_multi_stream4 = ExtractMostCommonFieldValues(
            stream_name="train",
            field="field",
            to_field="most_common_individuals4",
            overall_top_frequency_percent=90,
            process_every_value=True,
        ).process(input_multi_stream2)
        output_for_comparison4 = {}
        for k, v in output_multi_stream4.items():
            output_for_comparison4[k] = list(v)
        expected_output4 = {
            "test": [
                {
                    "field": ["a", "b", "c"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                    "most_common_individuals4": [
                        "h",
                        "m",
                        "i",
                        "j",
                        "k",
                        "o",
                        "p",
                        "q",
                        "r",
                        "s",
                    ],
                }
            ],
            "validation": [
                {
                    "field": ["d", "e", "f"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                    "most_common_individuals4": [
                        "h",
                        "m",
                        "i",
                        "j",
                        "k",
                        "o",
                        "p",
                        "q",
                        "r",
                        "s",
                    ],
                }
            ],
            "train": [
                {
                    "field": ["t", "u", "v"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                    "most_common_individuals4": [
                        "h",
                        "m",
                        "i",
                        "j",
                        "k",
                        "o",
                        "p",
                        "q",
                        "r",
                        "s",
                    ],
                },
                {
                    "field": ["h", "i", "j"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                    "most_common_individuals4": [
                        "h",
                        "m",
                        "i",
                        "j",
                        "k",
                        "o",
                        "p",
                        "q",
                        "r",
                        "s",
                    ],
                },
                {
                    "field": ["k", "h", "m"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                    "most_common_individuals4": [
                        "h",
                        "m",
                        "i",
                        "j",
                        "k",
                        "o",
                        "p",
                        "q",
                        "r",
                        "s",
                    ],
                },
                {
                    "field": ["m", "o", "p"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                    "most_common_individuals4": [
                        "h",
                        "m",
                        "i",
                        "j",
                        "k",
                        "o",
                        "p",
                        "q",
                        "r",
                        "s",
                    ],
                },
                {
                    "field": ["m", "o", "p"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                    "most_common_individuals4": [
                        "h",
                        "m",
                        "i",
                        "j",
                        "k",
                        "o",
                        "p",
                        "q",
                        "r",
                        "s",
                    ],
                },
                {
                    "field": ["h", "i", "j"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                    "most_common_individuals4": [
                        "h",
                        "m",
                        "i",
                        "j",
                        "k",
                        "o",
                        "p",
                        "q",
                        "r",
                        "s",
                    ],
                },
                {
                    "field": ["q", "r", "s"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                    "most_common_individuals4": [
                        "h",
                        "m",
                        "i",
                        "j",
                        "k",
                        "o",
                        "p",
                        "q",
                        "r",
                        "s",
                    ],
                },
                {
                    "field": ["k", "h", "m"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                    "most_common_individuals4": [
                        "h",
                        "m",
                        "i",
                        "j",
                        "k",
                        "o",
                        "p",
                        "q",
                        "r",
                        "s",
                    ],
                },
                {
                    "field": ["h", "i", "j"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                    "most_common_individuals4": [
                        "h",
                        "m",
                        "i",
                        "j",
                        "k",
                        "o",
                        "p",
                        "q",
                        "r",
                        "s",
                    ],
                },
                {
                    "field": ["q", "r", "s"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                    "most_common_individuals4": [
                        "h",
                        "m",
                        "i",
                        "j",
                        "k",
                        "o",
                        "p",
                        "q",
                        "r",
                        "s",
                    ],
                },
                {
                    "field": ["k", "h", "m"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                    "most_common_individuals4": [
                        "h",
                        "m",
                        "i",
                        "j",
                        "k",
                        "o",
                        "p",
                        "q",
                        "r",
                        "s",
                    ],
                },
                {
                    "field": ["m", "o", "p"],
                    "most_common_lists3": [
                        ["h", "i", "j"],
                        ["k", "h", "m"],
                        ["m", "o", "p"],
                        ["q", "r", "s"],
                    ],
                    "most_common_individuals4": [
                        "h",
                        "m",
                        "i",
                        "j",
                        "k",
                        "o",
                        "p",
                        "q",
                        "r",
                        "s",
                    ],
                },
            ],
        }
        self.assertDictEqual(
            output_for_comparison4,
            expected_output4,
            "expected to see: \n"
            + json.dumps(expected_output4)
            + "\n but instead, received: \n"
            + json.dumps(output_for_comparison4),
        )
        # test error cases
        with self.assertRaises(ValueError):
            ExtractMostCommonFieldValues(
                stream_name="train",
                field="animal",
                to_field="most_common_individuals",
                overall_top_frequency_percent=90,
                process_every_value=True,
            ).process(input_multi_stream1)

        with self.assertRaises(AssertionError):
            ExtractMostCommonFieldValues(
                stream_name="train",
                field="animal",
                to_field="most_common_individuals",
                overall_top_frequency_percent=90,
                min_frequency_percent=25,
            ).process(input_multi_stream1)
        with self.assertRaises(AssertionError):
            ExtractMostCommonFieldValues(
                stream_name="train",
                field="animal",
                to_field="most_common_individuals",
                overall_top_frequency_percent=120,
            ).process(input_multi_stream1)
        with self.assertRaises(AssertionError):
            ExtractMostCommonFieldValues(
                stream_name="train",
                field="animal",
                to_field="most_common_individuals",
                min_frequency_percent=-2,
            ).process(input_multi_stream1)

    def test_apply(self):
        in_instance = {"a": "lower"}
        operator = Apply("a", function=str.upper, to_field="b")
        st_from_upper = operator.function_to_str(str.upper)
        self.assertEqual(st_from_upper, "str.upper")
        upper_from_st = operator.str_to_function(st_from_upper)
        self.assertEqual("UPPER", upper_from_st("upper"))
        out_instance = operator.process(in_instance)
        self.assertDictEqual(out_instance, {"a": "lower", "b": "LOWER"})

        in_instance = {"a": ["input", "list"]}
        operator = Apply("a", function="tuple", to_field="b")
        st_from_tuple = operator.function_to_str(tuple)
        self.assertEqual(st_from_tuple, "builtins.tuple")
        tuple_from_st = operator.str_to_function("tuple")
        self.assertEqual((1, 2, 3), tuple_from_st([1, 2, 3]))
        out_instance = operator.process(in_instance)
        self.assertDictEqual(
            out_instance, {"a": ["input", "list"], "b": ("input", "list")}
        )

    def test_shuffle(self):
        inputs = [{"a": i} for i in range(15)]

        outputs = apply_operator(operator=Shuffle(page_size=10), inputs=inputs)

        inputs = [instance["a"] for instance in inputs]
        outputs = [instance["a"] for instance in outputs]

        self.assertNotEqual(inputs, outputs)
        self.assertListEqual(sorted(inputs), sorted(outputs))

        # test no mixing between pages:
        page_1_inputs = inputs[:10]
        page_2_inputs = inputs[10:]
        page_1_outputs = outputs[:10]
        page_2_outputs = outputs[10:]

        self.assertListEqual(sorted(page_1_inputs), sorted(page_1_outputs))
        self.assertListEqual(sorted(page_2_inputs), sorted(page_2_outputs))

        inputs_outputs_intersection = set(page_1_inputs).intersection(
            set(page_2_outputs)
        )
        self.assertSetEqual(inputs_outputs_intersection, set())

        inputs_outputs_intersection = set(page_2_inputs).intersection(
            set(page_1_outputs)
        )
        self.assertSetEqual(inputs_outputs_intersection, set())

    def test_balance(self):
        inputs = [
            {"a": 1, "b": "x"},
            {"a": 1, "b": "x"},
            {"a": 2, "b": "y"},
            {"a": 2, "b": "y"},
            {"a": 3, "b": "z"},
            {"a": 3, "b": "z"},
        ]
        targets = [
            {"a": 1, "b": "x"},
            {"a": 2, "b": "y"},
            {"a": 3, "b": "z"},
            {"a": 1, "b": "x"},
            {"a": 2, "b": "y"},
            {"a": 3, "b": "z"},
        ]

        check_operator(
            operator=Balance(by_fields=["a"]),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

        inputs = [{"a": 1, "b": "x"}] * 5
        targets = [{"a": 1, "b": "x"}] * 5

        check_operator(
            operator=Balance(by_fields=["a"]),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

        inputs = []
        targets = []

        check_operator(
            operator=Balance(by_fields=["a"]),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

        inputs = [
            {"a": 1, "b": "x"},
            {"a": 1, "b": "x"},
            {"a": 2, "b": "y"},
            {"a": 3, "b": "z"},
        ]
        targets = [
            {"a": 1, "b": "x"},
            {"a": 2, "b": "y"},
            {"a": 3, "b": "z"},
            {"a": 1, "b": "x"},
        ]

        check_operator(
            operator=Balance(by_fields=["a"]),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

        inputs = [
            {"a": 1, "b": "x"},
            {"a": "1", "b": "y"},
            {"a": 1.0, "b": "z"},
            {"a": None, "b": "w"},
        ]
        targets = inputs

        check_operator(
            operator=Balance(by_fields=["a"]),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_grouped_shuffle(self):
        # 18 questions indexed by "index", but with two grouping features: question, topic
        inputs = [
            {
                "question": 0,
                "topic": "biology",
                "type": "original",
                "text": "salamander",
                "index": 0,
            },
            {
                "question": 0,
                "topic": "biology",
                "type": "misspelling",
                "text": "slmnder",
                "index": 1,
            },
            {
                "question": 0,
                "topic": "biology",
                "type": "misspelling",
                "text": "salmnder",
                "index": 2,
            },
            {
                "question": 1,
                "topic": "biology",
                "type": "original",
                "text": "triceratops",
                "index": 3,
            },
            {
                "question": 1,
                "topic": "biology",
                "type": "misspelling",
                "text": "trceratps",
                "index": 4,
            },
            {
                "question": 1,
                "topic": "biology",
                "type": "misspelling",
                "text": "tricertp",
                "index": 5,
            },
            {
                "question": 2,
                "topic": "biology",
                "type": "original",
                "text": "monkey",
                "index": 6,
            },
            {
                "question": 2,
                "topic": "biology",
                "type": "misspelling",
                "text": "mnky",
                "index": 7,
            },
            {
                "question": 2,
                "topic": "biology",
                "type": "misspelling",
                "text": "monke",
                "index": 8,
            },
            {
                "question": 0,
                "topic": "history",
                "type": "original",
                "text": "George Washington",
                "index": 9,
            },
            {
                "question": 0,
                "topic": "history",
                "type": "misspelling",
                "text": "Gorg Washingtn",
                "index": 10,
            },
            {
                "question": 0,
                "topic": "history",
                "type": "misspelling",
                "text": "Gregory Washtn",
                "index": 11,
            },
            {
                "question": 1,
                "topic": "history",
                "type": "original",
                "text": "Marie Curie",
                "index": 12,
            },
            {
                "question": 1,
                "topic": "history",
                "type": "misspelling",
                "text": "Mare Cureu",
                "index": 13,
            },
            {
                "question": 1,
                "topic": "history",
                "type": "misspelling",
                "text": "Mary Curei",
                "index": 14,
            },
            {
                "question": 2,
                "topic": "history",
                "type": "original",
                "text": "Genghis Khan",
                "index": 15,
            },
            {
                "question": 2,
                "topic": "history",
                "type": "misspelling",
                "text": "Genghz Kanh",
                "index": 16,
            },
            {
                "question": 2,
                "topic": "history",
                "type": "misspelling",
                "text": "Gengz Kann",
                "index": 17,
            },
        ]

        outputs_1feature = apply_operator(
            operator=FeatureGroupedShuffle(grouping_features=["topic"], page_size=20),
            inputs=inputs,
        )
        outputs_2features_smallpage = apply_operator(
            operator=FeatureGroupedShuffle(
                grouping_features=["question", "topic"], page_size=10
            ),
            inputs=inputs,
        )
        # set the page size bigger than the list, so that the initial shuffle has access to the full groups for shuffling
        outputs_2features_bigpage = apply_operator(
            operator=FeatureGroupedShuffle(
                grouping_features=["question", "topic"], page_size=20
            ),
            inputs=inputs,
        )
        outputs_2features_shuffled = apply_operator(
            operator=FeatureGroupedShuffle(
                grouping_features=["question", "topic"],
                shuffle_within_group=True,
                page_size=20,
            ),
            inputs=inputs,
        )

        output_1feature_indices = [instance["index"] for instance in outputs_1feature]
        output_2features_indices_smallpage = [
            instance["index"] for instance in outputs_2features_smallpage
        ]
        # with a bigger page, expect to see more mixing across the indices for the full range
        output_2features_indices_bigpage = [
            instance["index"] for instance in outputs_2features_bigpage
        ]
        output_2features_shuffled_indices = [
            instance["index"] for instance in outputs_2features_shuffled
        ]

        # check the expected values
        self.assertListEqual(
            output_1feature_indices,
            [9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        )
        self.assertListEqual(
            output_2features_indices_smallpage,
            [9, 6, 7, 8, 3, 4, 5, 0, 1, 2, 12, 13, 14, 10, 11, 15, 16, 17],
        )
        self.assertListEqual(
            output_2features_indices_bigpage,
            [3, 4, 5, 15, 16, 17, 12, 13, 14, 6, 7, 8, 9, 10, 11, 0, 1, 2],
        )
        self.assertListEqual(
            output_2features_shuffled_indices,
            [3, 5, 4, 17, 16, 15, 13, 14, 12, 8, 6, 7, 10, 11, 9, 1, 0, 2],
        )

        # test no mixing between pages:
        input_indices = [instance["index"] for instance in inputs]
        page_1_inputs = input_indices[:10]
        page_2_inputs = input_indices[10:]
        page_1_outputs = output_2features_indices_smallpage[:10]
        page_2_outputs = output_2features_indices_smallpage[10:]

        self.assertListEqual(sorted(page_1_inputs), sorted(page_1_outputs))
        self.assertListEqual(sorted(page_2_inputs), sorted(page_2_outputs))

        inputs_outputs_intersection = set(page_1_inputs).intersection(
            set(page_2_outputs)
        )
        self.assertSetEqual(inputs_outputs_intersection, set())

        inputs_outputs_intersection = set(page_2_inputs).intersection(
            set(page_1_outputs)
        )
        self.assertSetEqual(inputs_outputs_intersection, set())

    def test_shuffle_field_value(self):
        operator = ShuffleFieldValues([["from", "to"]])
        in_list = [1, 2, 3, 4, 5, 6, 7, 8]
        out_list = operator.process_value(in_list)
        self.assertEqual(sorted(out_list), in_list)
        self.assertNotEqual(out_list, in_list)

    def test_cast_fields(self):
        inputs = [
            {"a": "0.5", "b": "2"},
            {"a": "fail", "b": "fail"},
        ]

        targets = [
            {"a": 0.5, "b": 2},
            {"a": 0.0, "b": 0},
        ]

        check_operator(
            operator=CastFields(
                fields={"a": "float", "b": "int"}, failure_defaults={"a": 0.0, "b": 0}
            ),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

        check_operator(
            operator=CastFields(
                fields={"a/d": "float", "b": "int"},
                failure_defaults={"a/d": 0.0, "b": 0},
                process_every_value=True,
                use_nested_query=True,
            ),
            inputs=[{"a": {"d": ["half", "0.6", 1, 12]}, "b": ["2"]}],
            targets=[{"a": {"d": [0.0, 0.6, 1.0, 12.0]}, "b": [2]}],
            tester=self,
        )

    def test_cast_fields_casting_failure(self):
        inputs = [
            {"a": "0.5", "b": "2"},
            {"a": "fail", "b": "fail"},
        ]

        with self.assertRaises(ValueError):
            apply_operator(
                operator=CastFields(fields={"a": "float", "b": "int"}), inputs=inputs
            )

    def test_divide_all_fields_by(self):
        instance_in = {"a": 10.0, "b": [2.0, 4.0, 7.0], "c": 5}
        operator = DivideAllFieldsBy(divisor=2.0)
        instance_out = operator.process(instance_in)
        expected = {"a": 5.0, "b": [1.0, 2.0, 3.5], "c": 5}
        (
            self.assertDictEqual(instance_out, expected),
            f"expected: {expected}, but got: {instance_out}.",
        )
        operator = DivideAllFieldsBy(
            divisor=2.0, strict=True
        )  # integer in "c" will raising ValueError with strict=False
        expected_error_message = "Cannot divide instance of type <class 'int'>"
        with self.assertRaises(ValueError) as ve:
            instance_out = operator.process(instance_in)
        (
            self.assertEqual(str(ve.exception), expected_error_message),
            f"expected error message: {expected_error_message}, but received: {ve.exception!s}.",
        )

    def test_rename(self):
        inputs = [
            {"a": 1, "b": 2},
            {"a": 2, "b": 3},
        ]

        targets = [
            {"a": 1, "c": 2},
            {"a": 2, "c": 3},
        ]

        # the simplest case
        check_operator(
            operator=Rename(field_to_field={"b": "c"}),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

        # to field is structured:
        check_operator(
            operator=Rename(field_to_field={"b": "c/d"}),
            inputs=inputs,
            targets=[{"a": 1, "c": {"d": 2}}, {"a": 2, "c": {"d": 3}}],
            tester=self,
        )

        # to field is structured, to stand in place of from field:
        check_operator(
            operator=Rename(field_to_field={"b": "b/d"}),
            inputs=inputs,
            targets=[{"a": 1, "b": {"d": 2}}, {"a": 2, "b": {"d": 3}}],
            tester=self,
        )

        # to field is structured, to stand in place of from field, from field is deeper:
        check_operator(
            operator=Rename(field_to_field={"b/c/e": "b/d"}),
            inputs=[
                {"a": 1, "b": {"c": {"e": 2, "f": 20}}},
                {"a": 2, "b": {"c": {"e": 3, "f": 30}}},
            ],
            targets=[
                {"a": 1, "b": {"c": {"f": 20}, "d": 2}},
                {"a": 2, "b": {"c": {"f": 30}, "d": 3}},
            ],
            tester=self,
        )

        # to field is structured, from field is structured too, different fields:
        check_operator(
            operator=Rename(field_to_field={"b/c/e": "g/h"}),
            inputs=[
                {"a": 1, "b": {"c": {"e": 2, "f": 20}}},
                {"a": 2, "b": {"c": {"e": 3, "f": 30}}},
            ],
            targets=[
                {"a": 1, "b": {"c": {"f": 20}}, "g": {"h": 2}},
                {"a": 2, "b": {"c": {"f": 30}}, "g": {"h": 3}},
            ],
            tester=self,
        )

        # both from and to field are structured, different only in the middle of the path:
        check_operator(
            operator=Rename(field_to_field={"a/b/c/d": "a/g/c/d"}),
            inputs=[
                {"a": {"b": {"c": {"d": {"e": 1}}}}, "b": 2},
            ],
            targets=[
                {"a": {"g": {"c": {"d": {"e": 1}}}}, "b": 2},
            ],
            tester=self,
        )

        # both from and to field are structured, different down the path:
        check_operator(
            operator=Rename(field_to_field={"a/b/c/d": "a/b/c/f"}),
            inputs=[
                {"a": {"b": {"c": {"d": {"e": 1}}}}, "b": 2},
            ],
            targets=[
                {"a": {"b": {"c": {"f": {"e": 1}}}}, "b": 2},
            ],
            tester=self,
        )

        with self.assertWarns(DeprecationWarning) as dw:
            Rename(field_to_field={"a/b/c/d": "a/b/c/f"}, use_query=True)
            self.assertEqual(
                "Field 'use_query' is deprecated. From now on, default behavior is compatible to use_query=True. Please remove this field from your code.",
                dw.warnings[0].message.args[0],
            )

    def test_add(self):
        check_operator(
            operator=AddConstant(field_to_field=[["a", "b"]], add=5),
            inputs=[{"a": 1}],
            targets=[{"a": 1, "b": 6}],
            tester=self,
        )

        check_operator(
            operator=AddConstant(
                field_to_field={"a": "b", "c": "d"}, add=5, process_every_value=True
            ),
            inputs=[{"a": [1, 2, 3], "c": [4, 5, 6]}],
            targets=[
                {"a": [1, 2, 3], "b": [6, 7, 8], "c": [4, 5, 6], "d": [9, 10, 11]}
            ],
            tester=self,
        )

        # test the loop in field_to_field, to be caught on init
        with self.assertRaises(AssertionError) as ae:
            AddConstant(
                field_to_field={"a": "b", "b": "a"}, add=15, process_every_value=True
            ).process_instance({"a": [1, 2, 3], "b": [11]})

        self.assertEqual(
            str(ae.exception),
            "In input argument 'field_to_field': {'a': 'b', 'b': 'a'}, field b is mapped to field a, while the latter is mapped to b. Whether b or a is processed first might impact end result.",
        )

        # test if two different from_field determine the same to_field, in field_to_field, to be caught on init
        with self.assertRaises(AssertionError) as ae:
            AddConstant(
                field_to_field={"a": "c", "b": "c"}, add=15, process_every_value=True
            ).process(instance={"a": [1, 2, 3], "b": [11]})

        self.assertEqual(
            str(ae.exception),
            "In input argument 'field_to_field': {'a': 'c', 'b': 'c'}, two different fields: a and b are mapped to field c. Whether a or b is processed last might impact end result.",
        )

        with self.assertRaises(ValueError) as ve:
            AddConstant(
                field_to_field={"a", "c", "b"}, add=15, process_every_value=True
            ).process(instance={"a": [1, 2, 3], "b": [11]})

        self.assertEqual(
            str(ve.exception),
            "Input argument 'field_to_field': {self.field_to_field} is neither of type List{List[str]] nor of type Dict[str, str].",
        )

    def test_copy_paste_fields(self):
        inputs = [
            {"a": [1, 3]},
            {"a": [2, 4]},
        ]

        targets = [{"a": 1}, {"a": 2}]

        check_operator(
            operator=Copy(field="a/0", to_field="a"),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_copy_string_to_string_error(self):
        inputs = [{"source": "hello", "task_data": 3}]

        targets = [{"source": "hello", "task_data": {"source": "hello"}}]

        check_operator(
            operator=Copy(field="source", to_field="task_data/source"),
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

        check_operator(
            operator=Copy(field="a", to_field="a/x"),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_label_encoder(self):
        inputs = [
            {"prediction": "red", "references": ["red", "blue"]},
            {"prediction": "blue", "references": ["blue"]},
            {"prediction": "green", "references": ["red"]},
            {"prediction": "", "references": ["red"]},
            {"prediction": "green", "references": []},
        ]

        targets = [
            {"prediction": 0, "references": [0, 1]},
            {"prediction": 1, "references": [1]},
            {"prediction": 2, "references": [0]},
            {"prediction": 3, "references": [0]},
            {"prediction": 2, "references": []},
        ]

        check_operator(
            operator=EncodeLabels(fields=["prediction", "references/*"]),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

        inputs = [{"prediction": "red", "references": "blue"}]
        exception_texts = [
            """Error processing instance '0' from stream 'test' in EncodeLabels due to the exception above.""",
            """query \"references/*\" did not match any item in dict:
prediction (str):
    red
references (str):
    blue
""",
        ]
        check_operator_exception(
            operator=EncodeLabels(fields=["references/*", "prediction"]),
            inputs=inputs,
            tester=self,
            exception_texts=exception_texts,
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

        check_operator(
            operator=JoinStr(field_to_field={"a": "b"}, separator=","),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_zip_fields(self):
        inputs = [
            {"a": [1, 3, 5], "b": [1, 3]},
            {"a": [2, 4, 6], "b": [2, 4]},
        ]

        targets = [
            {"a": [1, 3, 5], "b": [1, 3], "c": [(1, 1), (3, 3)]},
            {"a": [2, 4, 6], "b": [2, 4], "c": [(2, 2), (4, 4)]},
        ]

        targets_longest = [
            {"a": [1, 3, 5], "b": [1, 3], "c": [(1, 1), (3, 3), (5, None)]},
            {"a": [2, 4, 6], "b": [2, 4], "c": [(2, 2), (4, 4), (6, None)]},
        ]

        check_operator(
            operator=ZipFieldValues(fields=["a", "b"], to_field="c"),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

        check_operator(
            operator=ZipFieldValues(fields=["a", "b"], to_field="c", longest=True),
            inputs=inputs,
            targets=targets_longest,
            tester=self,
        )

    def test_index_of(self):
        operator = IndexOf(
            search_in="field_text", index_of="field_pattern", to_field="index"
        )
        in_instance = {
            "field_text": "the long story I was telling to everyone.",
            "field_pattern": "telling to",
        }
        out_instance = operator.process(in_instance)
        self.assertEqual(out_instance["index"], 21)

    def test_take_by_field(self):
        inputs = [
            {"a": [1, 3], "b": 0},
            {"a": {"a": 1}, "b": "a"},
        ]

        targets = [
            {"a": [1, 3], "b": 0, "c": 1},
            {"a": {"a": 1}, "b": "a", "c": 1},
        ]

        check_operator(
            operator=TakeByField(field="a", index="b", to_field="c"),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

        # field plays the role of to_field
        check_operator(
            operator=TakeByField(field="a", index="b"),
            inputs=inputs,
            targets=[{"a": 1, "b": 0}, {"a": 1, "b": "a"}],
            tester=self,
        )

    def test_stream_refiner(self):
        refiner = StreamRefiner(apply_to_streams=["train"], max_instances=1)

        ms = MultiStream.from_iterables(
            {"train": [{"x": 0}, {"x": 1}], "test": [{"x": 2}, {"x": 3}]}, copying=True
        )

        refined_ms = refiner(ms)

        train = list(refined_ms["train"])
        self.assertEqual(len(train), 1)

        test = list(refined_ms["test"])
        self.assertEqual(len(test), 2)

        refiner.max_instances = None
        refiner.apply_to_streams = ["test"]
        refined_refined_ms = refiner(refined_ms)
        test = list(refined_refined_ms["test"])
        self.assertEqual(len(test), 2)

    def test_deterministic_balancer_empty_stream(self):
        inputs = []

        targets = []

        check_operator(
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

        check_operator(
            operator=DeterministicBalancer(fields=["a", "b"], max_instances=2),
            inputs=inputs + inputs,
            targets=targets,
            tester=self,
        )

    def test_minimum_one_per_label_balancer(self):
        inputs = [
            {"text": "I love dogs", "label": "dogs", "id": 0},
            {"text": "I love dogs", "label": "dogs", "id": 1},
            {"text": "I love dogs", "label": "dogs", "id": 2},
            {"text": "I love dogs", "label": "dogs", "id": 3},
            {"text": "I love dogs", "label": "dogs", "id": 4},
            {"text": "I love dogs", "label": "dogs", "id": 5},
            {"text": "I love dogs", "label": "dogs", "id": 6},
            {"text": "I love dogs", "label": "dogs", "id": 7},
            {"text": "I love dogs", "label": "dogs", "id": 8},
            {"text": "I love cats", "label": "cats", "id": 9},
            {"text": "I love parrots", "label": "parrots", "id": 10},
        ]

        targets = [
            {"text": "I love cats", "label": "cats", "id": 9},
            {"text": "I love dogs", "label": "dogs", "id": 0},
            {"text": "I love parrots", "label": "parrots", "id": 10},
        ]

        check_operator(
            operator=MinimumOneExamplePerLabelRefiner(
                fields=["label"], max_instances=3
            ),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

        targets = [
            {"text": "I love dogs", "label": "dogs", "id": 0},
            {"text": "I love cats", "label": "cats", "id": 9},
            {"text": "I love parrots", "label": "parrots", "id": 10},
            {"text": "I love dogs", "label": "dogs", "id": 1},
        ]

        check_operator(
            operator=MinimumOneExamplePerLabelRefiner(
                fields=["label"], max_instances=4
            ),
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

        check_operator(
            operator=LengthBalancer(fields=["a"], segments_boundaries=[1]),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_duplicate_instance(self):
        inputs = [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
        ]

        targets = [
            {"a": 1, "b": 2},
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
            {"a": 3, "b": 4},
        ]

        check_operator(
            operator=DuplicateInstances(num_duplications=2),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_duplicate_instance_added_field(self):
        inputs = [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
        ]

        targets = [
            {"a": 1, "b": 2, "duplication_id": 0},
            {"a": 1, "b": 2, "duplication_id": 1},
            {"a": 3, "b": 4, "duplication_id": 0},
            {"a": 3, "b": 4, "duplication_id": 1},
        ]

        check_operator(
            operator=DuplicateInstances(
                num_duplications=2, duplication_index_field="duplication_id"
            ),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_collate_instance(self):
        inputs = [
            {"a": 1, "b": 2, "data_classification_policy": ["public"]},
            {"a": 2, "b": 2, "data_classification_policy": ["public"]},
            {"a": 3, "b": 2, "data_classification_policy": ["public"]},
            {"a": 4, "b": 2, "data_classification_policy": ["private"]},
            {"a": 5, "b": 2, "data_classification_policy": ["public"]},
        ]

        targets = [
            {"a": [1, 2], "b": [2, 2], "data_classification_policy": ["public"]},
            {
                "a": [3, 4],
                "b": [2, 2],
                "data_classification_policy": ["private", "public"],
            },
            {"a": [5], "b": [2], "data_classification_policy": ["public"]},
        ]

        check_operator(
            operator=CollateInstances(batch_size=2),
            inputs=inputs,
            targets=targets,
            tester=self,
        )


class TestApplyMetric(UnitxtTestCase):
    def _test_apply_metric(
        self,
        metrics,
        expected_score_name,
        expected_score_value,
        calc_confidence_intervals=False,
    ):
        inputs = [
            {
                "prediction": "0",
                "references": ["1"],
                "task_data": {"classes": ["0", "1", "2"]},
                "metrics": metrics,
            },
            {
                "prediction": "1",
                "references": ["1"],
                "task_data": {"classes": ["0", "1", "2"]},
                "metrics": metrics,
            },
            {
                "prediction": "0",
                "references": ["2"],
                "task_data": {"classes": ["0", "1", "2"]},
                "metrics": metrics,
            },
            {
                "prediction": "0",
                "references": ["0"],
                "task_data": {"classes": ["0", "1", "2"]},
                "metrics": metrics,
            },
        ]
        output = apply_operator(
            operator=ApplyMetric(
                metric_field="metrics",
                calc_confidence_intervals=calc_confidence_intervals,
            ),
            inputs=inputs,
        )
        global_metric_result = output[0]["score"]["global"]
        self.assertEqual(global_metric_result["score"], expected_score_value)
        self.assertEqual(global_metric_result["score_name"], expected_score_name)
        self.assertEqual(
            global_metric_result[expected_score_name], expected_score_value
        )
        self.assertEqual(
            "score_ci_low" in global_metric_result, calc_confidence_intervals
        )
        self.assertEqual(
            "score_ci_high" in global_metric_result, calc_confidence_intervals
        )
        return global_metric_result

    def test_apply_metric_with_empty_metric(self):
        """Test applying a metric for one metric, given as a string."""
        try:
            self._test_apply_metric(
                metrics="", expected_score_name="accuracy", expected_score_value=0.5
            )
        except Exception as e:
            self.assertIn(
                "Missing metric names in field 'metrics' and instance", str(e)
            )

    def test_apply_metric_with_single_string_metric(self):
        """Test applying a metric for one metric, given as a string."""
        self._test_apply_metric(
            metrics="metrics.accuracy",
            expected_score_name="accuracy",
            expected_score_value=0.5,
        )

    def test_apply_metric_with_confience_intervals(self):
        """Test applying a metric for one metric, given as a string."""
        self._test_apply_metric(
            metrics="metrics.accuracy",
            expected_score_name="accuracy",
            expected_score_value=0.5,
            calc_confidence_intervals=True,
        )

    def test_apply_metric_with_a_metric_pipeline_and_no_confidence_intervals(self):
        """Test applying a metric for one metric, given as a string.

        The metric here is a MetricPipeline.
        """
        self._test_apply_metric(
            metrics="metrics.squad", expected_score_name="f1", expected_score_value=0.5
        )

    def test_apply_metric_with_two_metrics_and_no_confidence_intervals(self):
        global_metric_result = self._test_apply_metric(
            metrics=["metrics.accuracy", "metrics.f1_macro"],
            expected_score_name="accuracy",
            expected_score_value=0.5,
        )
        # check that the second score is present too
        self.assertAlmostEqual(global_metric_result["f1_macro"], 0.388, delta=2)

    def test_apply_metric_with_two_identical_metrics(self):
        global_metric_result = self._test_apply_metric(
            metrics=["metrics.accuracy", "metrics.accuracy"],
            expected_score_name="accuracy",
            expected_score_value=0.5,
        )
        self.assertAlmostEqual(global_metric_result["accuracy"], 0.5, delta=2)

    def test_apply_metric_with_metrics_list(self):
        global_metric_result = self._test_apply_metric(
            metrics=[MetricsList(["metrics.accuracy", "metrics.f1_macro"])],
            expected_score_name="accuracy",
            expected_score_value=0.5,
        )
        self.assertAlmostEqual(global_metric_result["f1_macro"], 0.388, delta=2)

    def test_render_demonstrations(self):
        template = InputOutputTemplate(
            input_format='This is my sentence: "{text}"', output_format="{label}"
        )

        instance = {
            "demos": [
                {
                    "input_fields": {"text": "was so not good"},
                    "reference_fields": {"label": "negative"},
                },
                {
                    "input_fields": {"text": "was so good"},
                    "reference_fields": {"label": "positive"},
                },
            ]
        }

        demos_out = [template.process(demo_inst) for demo_inst in instance["demos"]]
        instance["demos"] = demos_out

        target = {
            "demos": [
                {
                    "input_fields": {"text": "was so not good"},
                    "reference_fields": {"label": "negative"},
                    "source": 'This is my sentence: "was so not good"',
                    "target": "negative",
                    "references": ["negative"],
                    "instruction": "",
                    "target_prefix": "",
                    "postprocessors": ["processors.to_string_stripped"],
                },
                {
                    "input_fields": {"text": "was so good"},
                    "reference_fields": {"label": "positive"},
                    "source": 'This is my sentence: "was so good"',
                    "target": "positive",
                    "references": ["positive"],
                    "instruction": "",
                    "target_prefix": "",
                    "postprocessors": ["processors.to_string_stripped"],
                },
            ]
        }

        self.assertDictEqual(instance, target)

    def test_render_demonstrations_multi_reference(self):
        template = MultiReferenceTemplate(
            input_format="This is my sentence: {text}", references_field="answer"
        )

        instance = {
            "demos": [
                {
                    "input_fields": {"text": "who was he?"},
                    "reference_fields": {"answer": ["Dan", "Yossi"]},
                },
                {
                    "input_fields": {"text": "who was she?"},
                    "reference_fields": {"answer": ["Shira", "Yael"]},
                },
            ]
        }

        demos_out = [template.process(demo_inst) for demo_inst in instance["demos"]]
        instance["demos"] = demos_out

        target = {
            "demos": [
                {
                    "input_fields": {"text": "who was he?"},
                    "reference_fields": {"answer": ["Dan", "Yossi"]},
                    "source": "This is my sentence: who was he?",
                    "target": "Dan",
                    "references": ["Dan", "Yossi"],
                    "instruction": "",
                    "target_prefix": "",
                    "postprocessors": ["processors.to_string_stripped"],
                },
                {
                    "input_fields": {"text": "who was she?"},
                    "reference_fields": {"answer": ["Shira", "Yael"]},
                    "source": "This is my sentence: who was she?",
                    "target": "Shira",
                    "references": ["Shira", "Yael"],
                    "instruction": "",
                    "target_prefix": "",
                    "postprocessors": ["processors.to_string_stripped"],
                },
            ]
        }

        self.assertDictEqual(instance, target)

    def test_icl_format_with_demonstrations(self):
        instance = {
            "source": "1+1",
            "target": "2",
            "instruction": "solve the math exercises",
            "input_fields": {},
        }
        demos_instances = [
            {"source": "1+2", "target": "3", "instruction": "solve the math exercises"},
            {"source": "4-2", "target": "2", "instruction": "solve the math exercises"},
        ]

        target = """Instruction:solve the math exercises

User:1+2
Agent:3

User:4-2
Agent:2

User:1+1
Agent:"""

        system_format = SystemFormat(
            demo_format="User:{source}\nAgent:{target}\n\n",
            model_input_format="Instruction:{instruction}\n\n{demos}User:{source}\nAgent:",
        )
        # refresh instance, from which icl_format popped the instruction, and add demos into it:
        instance["instruction"] = "solve the math exercises"
        instance["demos"] = demos_instances

        instance_out = system_format.process(instance)
        self.assertEqual(instance_out["source"], target)

    def test_system_format_with_demonstrations_and_instruction_after_demos(
        self,
    ):
        demo_instances = [
            {"source": "1+2", "target": "3", "instruction": "solve the math exercises"},
            {"source": "4-2", "target": "2", "instruction": "solve the math exercises"},
        ]
        instance = {
            "source": "1+1",
            "target": "2",
            "input_fields": {},
            "instruction": "solve the math exercises",
            "demos": demo_instances,
        }

        target = """User:1+2
Agent:3

User:4-2
Agent:2

User:solve the math exercises

1+1
Agent:"""
        system_format = SystemFormat(
            demo_format="User:{source}\nAgent:{target}\n\n",
            model_input_format="{demos}User:{instruction}\n\n{source}\nAgent:",
        )

        instance_out = system_format.process(instance)
        self.assertEqual(instance_out["source"], target)
        self.assertEqual(instance["source"], target)

    def test_system_format_without_demonstrations(self):
        instance = {
            "source": "1+1",
            "target": "2",
            "instruction": "solve the math exercises",
            "input_fields": {},
        }

        target = """Instruction:solve the math exercises

User:1+1
Agent:"""

        system_format = SystemFormat(
            demo_format="User:{source}\nAgent:{target}\n\n",
            model_input_format="Instruction:{instruction}\n\n{demos}User:{source}\nAgent:",
        )

        instance_out = system_format.process(instance)
        self.assertEqual(instance_out["source"], target)
        self.assertEqual(instance["source"], target)

    def test_model_input_formatter_without_demonstrations_or_instruction(self):
        instance = {"source": "1+1", "target": "2", "input_fields": {}}
        target = """User:1+1
Agent:"""

        system_format = SystemFormat(
            demo_format="User:{source}\nAgent:{target}\n\n",
            model_input_format="{instruction}{demos}User:{source}\nAgent:",
        )
        instance_out = system_format.process(instance)
        self.assertEqual(instance_out["source"], target)
        self.assertEqual(instance_out["source"], target)

    def test_system_format_without_demonstrations_and_empty_instruction(self):
        instance = {
            "source": "1+1",
            "target": "2",
            "instruction": "",
            "input_fields": {},
        }

        target = """User:1+1
Agent:"""

        system_format = SystemFormat(
            demo_format="User:{source}\nAgent:{target}\n\n",
            model_input_format="{instruction}{demos}User:{source}\nAgent:",
        )
        instance_out = system_format.process(instance)
        self.assertEqual(instance_out["source"], target)
        self.assertEqual(instance["source"], target)

    def test_perturb(self):
        instance = {
            "target": 1,
            "classes": [0, 1],
            "source": "Classify the given text to yes or no",
        }
        operator = Perturb(
            field="target", to_field="prediction", percentage_to_perturb=0
        )
        out = operator.process(instance)
        self.assertEqual(out["target"], out["prediction"])
        operator = Perturb(
            field="target",
            to_field="prediction",
            select_from=[0, 1],
            percentage_to_perturb=100,
        )
        predictions = []
        for _ in range(100):
            out = operator.process(instance)
            predictions.append(out["prediction"])
        counter = Counter(predictions)
        self.assertGreaterEqual(counter[0], 25)
        self.assertGreaterEqual(counter[1], 25)
        instance["target"] = "abcdefghijklmnop"
        operator = Perturb(
            field="target", to_field="prediction", percentage_to_perturb=100
        )
        out = operator.process(instance)
        self.assertGreater(len(out["target"]), len(out["prediction"]))
        instance["target"] = "a"
        out = operator.process(instance)
        self.assertEqual(out["target"], out["prediction"])
        instance["target"] = 10.0
        out = operator.process(instance)
        self.assertNotEqual(out["target"], out["prediction"])
        with self.assertRaises(AssertionError) as ae:
            operator = Perturb(field="target", percentage_to_perturb=200)
        self.assertEqual(
            "'percentage_to_perturb' should be in the range 0..100. Received 200",
            str(ae.exception),
        )

    def test_join_streams(self):
        input_multi_stream = MultiStream(
            {
                "questions": [
                    {
                        "question": "question_1",
                        "id": "1",
                        "data_classification_policy": ["public"],
                        "recipe_metadata": [],
                    },
                    {
                        "question": "question_2",
                        "id": "2",
                        "data_classification_policy": ["public"],
                        "recipe_metadata": [],
                    },
                ],
                "answers": [
                    {
                        "answer": "answer_1",
                        "id": "1",
                        "data_classification_policy": ["public"],
                        "recipe_metadata": [],
                    },
                    {
                        "answer": "answer_2",
                        "id": "2",
                        "data_classification_policy": ["public"],
                        "recipe_metadata": [],
                    },
                ],
                "train": [{"field": "train1"}],
            }
        )
        output_multi_stream = JoinStreams(
            left_stream="questions",
            right_stream="answers",
            how="inner",
            on=["id"],
            new_stream_name="questions_and_answers",
        )(input_multi_stream)
        self.assertListEqual(
            list(output_multi_stream.keys()),
            ["questions", "answers", "train", "questions_and_answers"],
        )
        joined_stream = list(output_multi_stream["questions_and_answers"])
        expected_joined_stream = [
            {
                "question": "question_1",
                "id": "1",
                "answer": "answer_1",
                "data_classification_policy": ["public"],
                "recipe_metadata": [],
            },
            {
                "question": "question_2",
                "id": "2",
                "answer": "answer_2",
                "data_classification_policy": ["public"],
                "recipe_metadata": [],
            },
        ]
        TestOperators().compare_streams(joined_stream, expected_joined_stream)

    def test_delete_split(self):
        input_multi_stream = MultiStream(
            {
                "train": [{"field": "train1"}],
                "dev": [{"field": "dev1"}],
                "test": [{"field": "test1"}],
            }
        )
        output_multi_stream = DeleteSplits(splits=["train", "dev"])(input_multi_stream)
        self.assertListEqual(list(output_multi_stream.keys()), ["test"])

    def test_filter_by_condition_based_on_fields(self):
        inputs = [{"question": "question_1", "id_1": "1", "id_2": "1"}]
        targets = [{"question": "question_1", "id_1": "1", "id_2": "1"}]
        check_operator(
            operator=FilterByConditionBasedOnFields(
                values={"id_1": "id_2"}, condition="eq"
            ),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_select_fields(self):
        input_multi_stream = MultiStream(
            {
                "questions": [
                    {
                        "question": "question_1",
                        "id_1": "1",
                        "id_2": "1",
                        "data_classification_policy": ["public"],
                        "recipe_metadata": [],
                    },
                ],
            }
        )
        output_multi_stream = SelectFields(fields=["question", "id_1"])(
            input_multi_stream
        )
        self.assertListEqual(list(output_multi_stream.keys()), ["questions"])
        joined_stream = list(output_multi_stream["questions"])
        expected_joined_stream = [
            {
                "question": "question_1",
                "id_1": "1",
                "data_classification_policy": ["public"],
                "recipe_metadata": [],
            }
        ]
        TestOperators().compare_streams(joined_stream, expected_joined_stream)
        
    def test_
