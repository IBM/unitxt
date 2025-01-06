from typing import Any, List

from unitxt.artifact import fetch_artifact
from unitxt.processors import MatchClosestOption, Substring
from unitxt.test_utils.operators import check_operator

from tests.utils import UnitxtTestCase


def list_to_stream_with_prediction_and_references(list: List[Any]) -> List[Any]:
    return [{"prediction": item, "references": [item]} for item in list]


def list_to_stream_with_prediction_or_references(
    list: List[Any], key: str
) -> List[Any]:
    if key == "prediction":
        stream = [{key: item} for item in list]
    elif key == "references":
        stream = [{key: [item]} for item in list]
    else:
        raise ValueError(
            f"Unexpected key, expected 'prediction' or 'references' - got {key}"
        )
    return stream


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
        inputs = [
            {"a": "correct"},
            {"a": "Not Sure"},
            {"a": "longer input"},
            {"a": "x"},
        ]
        targets1 = [
            {"a": "correct", "b": "or"},
            {"a": "Not Sure", "b": "ot"},
            {"a": "longer input", "b": "on"},
            {"a": "x", "b": ""},
        ]
        targets2 = [
            {"a": "correct", "b": "orrect"},
            {"a": "Not Sure", "b": "ot Sure"},
            {"a": "longer input", "b": "onger input"},
            {"a": "x", "b": ""},
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

    def test_to_list_by_hyphen_space(self):
        parser, _ = fetch_artifact("processors.to_list_by_hyphen_space")
        inputs = ["- cat\n- dog", "- man\n- woman\n- dog"]
        targets = [["cat", "dog"], ["man", "woman", "dog"]]

        check_operator(
            operator=parser,
            inputs=list_to_stream_with_prediction_and_references(inputs),
            targets=list_to_stream_with_prediction_and_references(targets),
            tester=self,
        )

    def test_select_closest_option(self):
        inputs = [
            {
                "prediction": "option2",
                "task_data": {"options": ["option1", "choice2", "cheese3"]},
            },
            {
                "prediction": "choice1",
                "task_data": {"options": ["option1", "choice2", "cheese3"]},
            },
            {
                "prediction": "chase3",
                "task_data": {"options": ["option1", "choice2", "cheese3"]},
            },
            {
                "prediction": "2",
                "task_data": {"options": ["option1", "choice2", "cheese3"]},
            },
        ]

        targets = [
            {
                "prediction": "option1",
                "task_data": {"options": ["option1", "choice2", "cheese3"]},
            },
            {
                "prediction": "choice2",
                "task_data": {"options": ["option1", "choice2", "cheese3"]},
            },
            {
                "prediction": "cheese3",
                "task_data": {"options": ["option1", "choice2", "cheese3"]},
            },
            {
                "prediction": "choice2",
                "task_data": {"options": ["option1", "choice2", "cheese3"]},
            },
        ]

        check_operator(
            operator=MatchClosestOption(field="prediction"),
            inputs=inputs,
            targets=targets,
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

    def test_predictions_yes_1_else_0(self):
        parser, _ = fetch_artifact("processors.predictions_yes_1_else_0")
        inputs = ["yes", "no", "yaa"]
        targets = [
            {"references": ["yes"], "prediction": "1"},
            {"references": ["no"], "prediction": "0"},
            {"references": ["yaa"], "prediction": "0"},
        ]

        check_operator(
            operator=parser,
            inputs=list_to_stream_with_prediction_and_references(inputs),
            targets=targets,
            tester=self,
        )

    def test_str_to_float_format(self):
        parser, _ = fetch_artifact("processors.str_to_float_format")
        inputs = ["-2.4", "5", "5a"]
        targets = ["-2.4", "5.0", "5a"]

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

    def test_load_json_from_predictions(self):
        parser, _ = fetch_artifact("processors.load_json_from_predictions")
        inputs = [{"prediction": '{"yes":0.46}', "references": '{"yes":0.46}'}]
        targets = [{"prediction": {"yes": 0.46}, "references": '{"yes":0.46}'}]
        check_operator(
            operator=parser,
            inputs=inputs,
            targets=targets,
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

    def test_extract_mt_bench_rating_judgment(self):
        postprocessor, _ = fetch_artifact("processors.extract_mt_bench_rating_judgment")
        predictions = [
            "no reason 3.14 [[3]]",
            "[[6]]",
            "[[6.2]]",
            "[[9]] because",
            "good",
            "bad [[x]]",
        ]
        targets = [0.3, 0.6, 0.62, 0.9, 0.0, 0.0]

        check_operator(
            operator=postprocessor,
            inputs=list_to_stream_with_prediction_and_references(predictions),
            targets=list_to_stream_with_prediction_and_references(targets),
            tester=self,
        )

    def test_extract_mt_bench_label_judgment(self):
        postprocessor, _ = fetch_artifact("processors.extract_mt_bench_label_judgment")
        predictions = [
            "no reason 3.14 [[A]]",
            "[[B]]",
            "[[A]] because",
            "good",
            "bad [[C]]",
        ]
        targets = ["choice_a", "choice_b", "choice_a", "None", "tie"]

        check_operator(
            operator=postprocessor,
            inputs=list_to_stream_with_prediction_and_references(predictions),
            targets=list_to_stream_with_prediction_and_references(targets),
            tester=self,
        )

    def test_literal_eval(self):
        parser, _ = fetch_artifact("processors.literal_eval")
        inputs = [
            "[1, 2, 3]",
            "[1,2,3]",
            " [1,2,3] ",
            "['bob','sam','lance']",
            "[{'name': 'bob', 'age': 42}, {'name': 'Stacy', 'age': 25}]",
            "{'name': 'Stacy', 'age': 25}",
            "{'names': ['Sam', 'Bob'], 'age': 25}",
        ]
        targets = [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            ["bob", "sam", "lance"],
            [{"name": "bob", "age": 42}, {"name": "Stacy", "age": 25}],
            {"name": "Stacy", "age": 25},
            {"names": ["Sam", "Bob"], "age": 25},
        ]

        check_operator(
            operator=parser,
            inputs=list_to_stream_with_prediction_or_references(
                inputs, key="prediction"
            ),
            targets=list_to_stream_with_prediction_or_references(
                targets, key="prediction"
            ),
            tester=self,
        )
        parser, _ = fetch_artifact(
            "processors.literal_eval[process_references=True, process_prediction=False]"
        )
        check_operator(
            operator=parser,
            inputs=list_to_stream_with_prediction_or_references(
                inputs, key="references"
            ),
            targets=list_to_stream_with_prediction_or_references(
                targets, key="references"
            ),
            tester=self,
        )
