from typing import Any, List

from src.unitxt.artifact import fetch_artifact
from src.unitxt.test_utils.operators import check_operator
from src.unitxt.processors import Substring
from tests.utils import UnitxtTestCase


def list_to_stream_with_prediction_and_references(list: List[Any]) -> List[Any]:
    return [{"prediction": item, "references": [item]} for item in list]


class TestPostProcessors(UnitxtTestCase):
    def test_convert_to_boolean(self):
        parser, _ = fetch_artifact("processors.convert_to_boolean")
        inputs = [
            "that's right",
            "correct",
            "not sure",
            "true",
            "TRUE",
            "false",
            "interesting",
        ]
        targets = ["TRUE", "TRUE", "FALSE", "TRUE", "TRUE", "FALSE", "OTHER"]

        check_operator(
            operator=parser,
            inputs=list_to_stream_with_prediction_and_references(inputs),
            targets=list_to_stream_with_prediction_and_references(targets),
            tester=self,
        )

    def test_lower_case(self):
        parser, _ = fetch_artifact("processors.lower_case")
        inputs = [
            "correct",
            "Not Sure",
            "TRUE",
        ]
        targets = ["correct", "not sure", "true"]

        check_operator(
            operator=parser,
            inputs=list_to_stream_with_prediction_and_references(inputs),
            targets=list_to_stream_with_prediction_and_references(targets),
            tester=self,
        )

    def test_capitalize(self):
        parser, _ = fetch_artifact("processors.capitalize")
        inputs = [
            "correct",
            "Not Sure",
            "TRUE",
            "wORD",
        ]
        targets = ["Correct", "Not sure", "True", "Word"]

        check_operator(
            operator=parser,
            inputs=list_to_stream_with_prediction_and_references(inputs),
            targets=list_to_stream_with_prediction_and_references(targets),
            tester=self,
        )

    def test_substring(self):
        #parser, _ = fetch_artifact("processors.substring")
        inputs = [
            {"a" : "correct"},
            {"a" : "Not Sure"},
            {"a" : "longer input"},
            {"a" : "x"},
        ]
        targets1 = [
            {"a" : "correct", "b" : "or"},
            {"a" : "Not Sure", "b" : "ot"},
            {"a" : "longer input", "b" : "on"},
            {"a" : "x", "b" : ""},
        ]
        targets2 = [
            {"a" : "correct", "b" : "orrect"},
            {"a" : "Not Sure", "b" : "ot Sure"},
            {"a" : "longer input", "b" : "onger input"},
            {"a" : "x", "b" : ""},
        ]

        check_operator(
            operator=Substring(field="a", to_field="b", begin=1, end=3),
            inputs=inputs,
            targets=targets1,
            tester=self,
        )
        check_operator(
            operator=Substring(field="a", to_field="b", begin=1),
            inputs=inputs,
            targets=targets2,
            tester=self,
        )

    def test_to_span_label_pairs(self):
        parser, _ = fetch_artifact("processors.to_span_label_pairs")
        inputs = [r"John\,\: Doe: PER, New York: LOC, Goo\:gle: ORG", "None"]
        targets = [
            [("John\\,\\: Doe", "PER"), ("New York", "LOC"), ("Goo\\:gle", "ORG")],
            [],
        ]

        check_operator(
            operator=parser,
            inputs=list_to_stream_with_prediction_and_references(inputs),
            targets=list_to_stream_with_prediction_and_references(targets),
            tester=self,
        )

    def test_to_list_by_comma(self):
        parser, _ = fetch_artifact("processors.to_list_by_comma")
        inputs = ["cat, dog", "man, woman, dog"]
        targets = [["cat", "dog"], ["man", "woman", "dog"]]

        check_operator(
            operator=parser,
            inputs=list_to_stream_with_prediction_and_references(inputs),
            targets=list_to_stream_with_prediction_and_references(targets),
            tester=self,
        )

    def test_take_first_word(self):
        parser, _ = fetch_artifact("processors.take_first_word")
        inputs = ["- yes, I think it is"]
        targets = ["yes"]

        check_operator(
            operator=parser,
            inputs=list_to_stream_with_prediction_and_references(inputs),
            targets=list_to_stream_with_prediction_and_references(targets),
            tester=self,
        )
        inputs = ["..."]
        targets = [""]

        check_operator(
            operator=parser,
            inputs=list_to_stream_with_prediction_and_references(inputs),
            targets=list_to_stream_with_prediction_and_references(targets),
            tester=self,
        )

    def test_yes_no_to_int(self):
        parser, _ = fetch_artifact("processors.yes_no_to_int")
        inputs = ["yes", "no", "yaa"]
        targets = ["1", "0", "yaa"]

        check_operator(
            operator=parser,
            inputs=list_to_stream_with_prediction_and_references(inputs),
            targets=list_to_stream_with_prediction_and_references(targets),
            tester=self,
        )

    def test_to_yes_or_none(self):
        parser, _ = fetch_artifact("processors.to_yes_or_none")
        inputs = ["yes", "no", "yaa"]
        targets = ["yes", "none", "none"]

        check_operator(
            operator=parser,
            inputs=list_to_stream_with_prediction_and_references(inputs),
            targets=list_to_stream_with_prediction_and_references(targets),
            tester=self,
        )

    def test_stance_to_pro_con(self):
        parser, _ = fetch_artifact("processors.stance_to_pro_con")
        inputs = ["positive", "negative", "suggestion", "neutral", "nothing"]
        targets = ["PRO", "CON", "CON", "none", "none"]

        check_operator(
            operator=parser,
            inputs=list_to_stream_with_prediction_and_references(inputs),
            targets=list_to_stream_with_prediction_and_references(targets),
            tester=self,
        )

    def test_to_span_label_pairs_surface_only(self):
        parser, _ = fetch_artifact("processors.to_span_label_pairs_surface_only")
        inputs = [r"John\,\: Doe, New York", "None"]
        targets = [[("John\\,\\: Doe", ""), ("New York", "")], []]

        check_operator(
            operator=parser,
            inputs=list_to_stream_with_prediction_and_references(inputs),
            targets=list_to_stream_with_prediction_and_references(targets),
            tester=self,
        )

    def test_load_json(self):
        parser, _ = fetch_artifact("processors.load_json")
        inputs = [
            '{"PER": ["John,: Doe", "New York"], "ORG": ["Goo:gle"]}',
            "None",
        ]

        targets = [
            {"PER": ["John,: Doe", "New York"], "ORG": ["Goo:gle"]},
            [],
        ]

        check_operator(
            operator=parser,
            inputs=list_to_stream_with_prediction_and_references(inputs),
            targets=list_to_stream_with_prediction_and_references(targets),
            tester=self,
        )

    def test_dict_of_lists_to_value_key_pairs(self):
        parser, _ = fetch_artifact("processors.dict_of_lists_to_value_key_pairs")
        inputs = [
            {"PER": ["John,: Doe", "New York"], "ORG": ["Goo:gle"]},
            {},
        ]

        targets = [
            [("John,: Doe", "PER"), ("New York", "PER"), ("Goo:gle", "ORG")],
            [],
        ]

        check_operator(
            operator=parser,
            inputs=list_to_stream_with_prediction_and_references(inputs),
            targets=list_to_stream_with_prediction_and_references(targets),
            tester=self,
        )

    def test_span_labeling_json_template_errors(self):
        postprocessor1, _ = fetch_artifact("processors.load_json")
        postprocessor2, _ = fetch_artifact(
            "processors.dict_of_lists_to_value_key_pairs"
        )

        predictions = ["{}", '{"d":{"b": "c"}}', '{dll:"dkk"}', '["djje", "djjjd"]']

        post1_targets = [{}, {"d": {"b": "c"}}, [], ["djje", "djjjd"]]
        post2_targets = [[], [("b", "d")], [], []]

        check_operator(
            operator=postprocessor1,
            inputs=list_to_stream_with_prediction_and_references(predictions),
            targets=list_to_stream_with_prediction_and_references(post1_targets),
            tester=self,
        )
        check_operator(
            operator=postprocessor2,
            inputs=list_to_stream_with_prediction_and_references(post1_targets),
            targets=list_to_stream_with_prediction_and_references(post2_targets),
            tester=self,
        )
