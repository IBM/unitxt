from math import isnan
from typing import Dict, List

from unitxt.api import create_dataset, evaluate
from unitxt.inference import MockInferenceEngine
from unitxt.llm_as_judge import LLMAsJudge, TaskBasedLLMasJudge
from unitxt.logging_utils import get_logger
from unitxt.metrics import (
    ANLS,
    NER,
    Accuracy,
    AccuracyFast,
    BinaryAccuracy,
    BinaryMaxAccuracy,
    BinaryMaxF1,
    CharEditDistance,
    CharEditDistanceAccuracy,
    Detector,
    F1Binary,
    F1BinaryPosOnly,
    F1Fast,
    F1Macro,
    F1MacroMultiLabel,
    F1Micro,
    F1MicroMultiLabel,
    F1Strings,
    F1Weighted,
    FinQAEval,
    FixedGroupAbsvalNormCohensHParaphraseAccuracy,
    FixedGroupAbsvalNormCohensHParaphraseStringContainment,
    FixedGroupAbsvalNormHedgesGParaphraseAccuracy,
    FixedGroupAbsvalNormHedgesGParaphraseStringContainment,
    FixedGroupMeanAccuracy,
    FixedGroupMeanBaselineAccuracy,
    FixedGroupMeanBaselineStringContainment,
    FixedGroupMeanParaphraseAccuracy,
    FixedGroupMeanParaphraseStringContainment,
    FixedGroupMeanStringContainment,
    FixedGroupNormCohensHParaphraseAccuracy,
    FixedGroupNormCohensHParaphraseStringContainment,
    FixedGroupNormHedgesGParaphraseAccuracy,
    FixedGroupNormHedgesGParaphraseStringContainment,
    FixedGroupPDRParaphraseAccuracy,
    FixedGroupPDRParaphraseStringContainment,
    FuzzyNer,
    GroupMeanAccuracy,
    GroupMeanStringContainment,
    GroupMeanTokenOverlap,
    HuggingfaceMetric,
    KendallTauMetric,
    KeyValueExtraction,
    LlamaIndexCorrectness,
    MaxAccuracy,
    MeteorFast,
    MetricsEnsemble,
    NormalizedSacrebleu,
    Perplexity,
    PrecisionBinary,
    RecallBinary,
    RelaxedCorrectness,
    RocAuc,
    Rouge,
    SQLExecutionAccuracy,
    SQLNonExecutionAccuracy,
    StringContainment,
    StringContainmentRatio,
    TokenOverlap,
    ToolCallingMetric,
    UnsortedListExactMatch,
    WebsrcSquadF1,
)
from unitxt.test_utils.metrics import (
    apply_metric,
    check_scores,
    test_metric,
)

from tests.utils import UnitxtTestCase

logger = get_logger()

# values of inputs that are common to grouped_mean type InstanceMetric
GROUPED_INSTANCE_PREDICTIONS = [
    "A B",
    "BC D",
    "C",
    "123",
    "BCD",
    "10",
    "  BD",
    "AB",
    "I am a dog",
    "AB C",
    "AB 1",
    "GMA",
    "0.123",
    "BD",
    "abc",
]

GROUPED_INSTANCE_REFERENCES = [
    ["B", "AB", "A"],
    ["A", "BC D", "BC DF"],
    ["c", " C"],
    ["13", "23", "234"],
    ["  ", " BD", " BDA"],
    ["1", "10", "100"],
    ["A", "B", "BD"],
    ["ABC", "ab", "BC"],
    ["I am a person", "I AM A DOG", "ABC"],
    ["AB CD", "AB", "ab"],
    ["AB 1", "AB1"],
    [" GMA 123", "GMA"],
    ["123", "0.12"],
    ["BDE", "BCE", "bdefs"],
    [" abcdefg", "AB", "abcd"],
]

# task_data, consisting of a group_id (group instance scores by this, then apply aggregation function)
# and variant_type (for metrics that compare, say original vs paraphrase instance score)
# create 4 groups, of sizes 5,5,4,1
GROUPED_INSTANCE_ADDL_INPUTS = [
    {"group_id": "group1", "variant_type": "original"},
    {"group_id": "group1", "variant_type": "paraphrase"},
    {"group_id": "group1", "variant_type": "paraphrase"},
    {"group_id": "group1", "variant_type": "paraphrase"},
    {"group_id": "group1", "variant_type": "paraphrase"},
    {"group_id": "group2", "variant_type": "original"},
    {"group_id": "group2", "variant_type": "paraphrase"},
    {"group_id": "group2", "variant_type": "paraphrase"},
    {"group_id": "group2", "variant_type": "paraphrase"},
    {"group_id": "group2", "variant_type": "paraphrase"},
    {"group_id": "group3", "variant_type": "original"},
    {"group_id": "group3", "variant_type": "paraphrase"},
    {"group_id": "group3", "variant_type": "paraphrase"},
    {"group_id": "group3", "variant_type": "paraphrase"},
    {"group_id": "group4", "variant_type": "original"},
]


class TestMetrics(UnitxtTestCase):
    def test_unsorted_list_exact_match(self):
        metric = UnsortedListExactMatch()

        predictions = [["A", "B"], ["B", "A"], ["A", "B", "C"]]
        references = [[["A", "B"]], [["A", "B"]], [["A", "B", "D"]]]

        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )

        expected_global_result = {
            "unsorted_list_exact_match": 2 / 3,
            "score": 2 / 3,
            "score_name": "unsorted_list_exact_match",
        }

        global_result = outputs[0]["score"]["global"].copy()
        # Only check the keys that are expected, i.e. exist in expected_global_result
        global_result = {
            key: value
            for key, value in global_result.items()
            if key in expected_global_result
        }
        self.assertDictEqual(global_result, expected_global_result)

        instance_targets = [
            {
                "unsorted_list_exact_match": 1.0,
                "score": 1.0,
                "score_name": "unsorted_list_exact_match",
            },
            {
                "unsorted_list_exact_match": 1.0,
                "score": 1.0,
                "score_name": "unsorted_list_exact_match",
            },
            {
                "unsorted_list_exact_match": 0.0,
                "score": 0.0,
                "score_name": "unsorted_list_exact_match",
            },
        ]
        for output, target in zip(outputs, instance_targets):
            self.assertDictEqual(output["score"]["instance"], target)

    def prediction_type_definition(self):
        class TempAccuracy(Accuracy):
            prediction_type = int

        self.assertEqual(TempAccuracy().prediction_type, int)

    def test_prediction_type_definition_deprecated(self):
        class TempAccuracy2(Accuracy):
            prediction_type = "int"

        self.assertEqual(TempAccuracy2().prediction_type, int)

    def test_accuracy(self):
        metric = Accuracy()

        predictions = ["A", "B", "C"]
        references = [["B", "C"], ["A"], ["B", "C"]]

        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )

        expected_global_result = {
            "accuracy": 1 / 3,
            "score": 1 / 3,
            "score_name": "accuracy",
        }

        global_result = outputs[0]["score"]["global"].copy()
        # Only check the keys that are expected, i.e. exist in expected_global_result
        global_result = {
            key: value
            for key, value in global_result.items()
            if key in expected_global_result
        }
        self.assertDictEqual(global_result, expected_global_result)

        instance_targets = [
            {"accuracy": 0.0, "score": 0.0, "score_name": "accuracy"},
            {"accuracy": 0.0, "score": 0.0, "score_name": "accuracy"},
            {"accuracy": 1.0, "score": 1.0, "score_name": "accuracy"},
        ]
        for output, target in zip(outputs, instance_targets):
            self.assertDictEqual(output["score"]["instance"], target)

    def test_accuracy_fast(self):
        metric = AccuracyFast()

        predictions = ["A", "B", "C"]
        references = [["B", "C"], ["A"], ["B", "C"]]

        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )

        expected_global_result = {
            "accuracy": 1 / 3,
            "score": 1 / 3,
            "score_name": "accuracy",
        }

        global_result = outputs[0]["score"]["global"].copy()
        # Only check the keys that are expected, i.e. exist in expected_global_result
        global_result = {
            key: value
            for key, value in global_result.items()
            if key in expected_global_result
        }
        self.assertDictEqual(global_result, expected_global_result)

        instance_targets = [
            {"accuracy": 0.0, "score": 0.0, "score_name": "accuracy"},
            {"accuracy": 0.0, "score": 0.0, "score_name": "accuracy"},
            {"accuracy": 1.0, "score": 1.0, "score_name": "accuracy"},
        ]
        for output, target in zip(outputs, instance_targets):
            self.assertDictEqual(output["score"]["instance"], target)

    def test_accuracy_with_prefix(self):
        metric = Accuracy(score_prefix="my_")

        predictions = ["A", "B", "C"]
        references = [["B", "C"], ["A"], ["B", "C"]]

        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )

        expected_global_result = {
            "my_accuracy": 1 / 3,
            "score": 1 / 3,
            "score_name": "my_accuracy",
        }

        global_result = outputs[0]["score"]["global"].copy()
        # Only check the keys that are expected, i.e. exist in expected_global_result
        global_result = {
            key: value
            for key, value in global_result.items()
            if key in expected_global_result
        }
        self.assertDictEqual(global_result, expected_global_result)

        instance_targets = [
            {"my_accuracy": 0.0, "score": 0.0, "score_name": "my_accuracy"},
            {"my_accuracy": 0.0, "score": 0.0, "score_name": "my_accuracy"},
            {"my_accuracy": 1.0, "score": 1.0, "score_name": "my_accuracy"},
        ]
        for output, target in zip(outputs, instance_targets):
            self.assertDictEqual(output["score"]["instance"], target)

    def test_accuracy_max_aggregation(self):
        metric = MaxAccuracy()

        predictions = ["A", "B", "C"]
        references = [["B", "C"], ["A"], ["B", "C"]]

        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )

        expected_global_result = {
            "accuracy": 1,
            "score": 1,
            "score_name": "accuracy",
        }

        global_result = outputs[0]["score"]["global"].copy()
        # Only check the keys that are expected, i.e. exist in expected_global_result
        global_result = {
            key: value
            for key, value in global_result.items()
            if key in expected_global_result
        }
        self.assertDictEqual(global_result, expected_global_result)

        instance_targets = [
            {"accuracy": 0.0, "score": 0.0, "score_name": "accuracy"},
            {"accuracy": 0.0, "score": 0.0, "score_name": "accuracy"},
            {"accuracy": 1.0, "score": 1.0, "score_name": "accuracy"},
        ]
        for output, target in zip(outputs, instance_targets):
            self.assertDictEqual(output["score"]["instance"], target)

    def test_f1_micro(self):
        metric = F1Micro()
        references = [["cat"], ["dog"], ["dog"], ["dog"], ["cat"], ["cat"]]
        predictions = ["cat", "cat", "dog", "dog", "cat", "cat"]
        # F1 micro is equal to accuracy in multi class setting  (5/6)
        global_target = 0.8333333
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        self.assertEqual("f1_micro", outputs[0]["score"]["global"]["score_name"])
        self.assertEqual("f1_micro", outputs[0]["score"]["instance"]["score_name"])

    def test_f1_strings(self):
        metric = F1Strings()
        references = [["cat dog"], ["dog"], ["cat"], ["cat"], ["cat"], ["gfjgfh"]]
        predictions = ["cat", "dog", "dog", "dog cat.", "dog Cat mouse", "100,000"]

        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )

        instance_targets = [
            {"f1_strings": [2 / 3], "score": [2 / 3], "score_name": "f1_strings"},
            {"f1_strings": [1.0], "score": [1.0], "score_name": "f1_strings"},
            {"f1_strings": [0.0], "score": [0.0], "score_name": "f1_strings"},
            {"f1_strings": [0.5], "score": [0.5], "score_name": "f1_strings"},
            {"f1_strings": [0.5], "score": [0.5], "score_name": "f1_strings"},
        ]
        for output, target in zip(outputs, instance_targets):
            self.assertDictEqual(output["score"]["instance"], target)

    def test_f1_micro_with_prefix(self):
        metric = F1Micro(score_prefix="my_")

        references = [["cat"], ["dog"], ["dog"], ["dog"], ["cat"], ["cat"]]
        predictions = ["cat", "cat", "dog", "dog", "cat", "cat"]

        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )

        expected_global_result = {
            "my_f1_micro": 5 / 6,
            "score": 5 / 6,
            "score_name": "my_f1_micro",
        }

        global_result = outputs[0]["score"]["global"].copy()
        # Only check the keys that are expected, i.e. exist in expected_global_result
        global_result = {
            key: value
            for key, value in global_result.items()
            if key in expected_global_result
        }
        self.assertDictEqual(global_result, expected_global_result)

        instance_targets = [
            {"my_f1_micro": 1.0, "score": 1.0, "score_name": "my_f1_micro"},
            {"my_f1_micro": 0.0, "score": 0.0, "score_name": "my_f1_micro"},
            {"my_f1_micro": 1.0, "score": 1.0, "score_name": "my_f1_micro"},
            {"my_f1_micro": 1.0, "score": 1.0, "score_name": "my_f1_micro"},
            {"my_f1_micro": 1.0, "score": 1.0, "score_name": "my_f1_micro"},
            {"my_f1_micro": 1.0, "score": 1.0, "score_name": "my_f1_micro"},
        ]
        for output, target in zip(outputs, instance_targets):
            self.assertDictEqual(output["score"]["instance"], target)

    def test_f1_micro_map_reduce_with_prefix(self):
        metric = F1Fast(main_score="f1_micro", averages=["micro"], score_prefix="my_")

        references = [["cat"], ["dog"], ["dog"], ["dog"], ["cat"], ["cat"]]
        predictions = ["cat", "cat", "dog", "dog", "cat", "cat"]

        outputs = apply_metric(
            metric=metric,
            predictions=predictions,
            references=references,
        )

        expected_global_result = {
            "my_f1_micro": 5 / 6,
            "score": 5 / 6,
            "score_name": "my_f1_micro",
        }

        global_result = outputs[0]["score"]["global"].copy()
        # Only check the keys that are expected, i.e. exist in expected_global_result
        global_result = {
            key: value
            for key, value in global_result.items()
            if key in expected_global_result
        }
        self.assertDictEqual(global_result, expected_global_result)

        instance_targets = [
            {"my_f1_micro": 1.0, "score": 1.0, "score_name": "my_f1_micro"},
            {"my_f1_micro": 0.0, "score": 0.0, "score_name": "my_f1_micro"},
            {"my_f1_micro": 1.0, "score": 1.0, "score_name": "my_f1_micro"},
            {"my_f1_micro": 1.0, "score": 1.0, "score_name": "my_f1_micro"},
            {"my_f1_micro": 1.0, "score": 1.0, "score_name": "my_f1_micro"},
            {"my_f1_micro": 1.0, "score": 1.0, "score_name": "my_f1_micro"},
        ]
        for output, target in zip(outputs, instance_targets):
            self.assertDictEqual(output["score"]["instance"], target)

    def test_f1_errors(self):
        metric = F1Micro()

        references = [["cat"]]
        predictions = [None]
        with self.assertRaises(ValueError) as cm:
            apply_metric(metric=metric, predictions=predictions, references=references)
        self.assertEqual(
            str(cm.exception),
            "Each prediction is expected to be of type 'str' in F1Micro metric. Received prediction of type <class 'NoneType'>: None",
        )

        references = [["cat"], "dog"]
        predictions = ["cat", "dog"]
        with self.assertRaises(ValueError) as cm:
            # disable validationd done in apply_metric
            apply_metric(
                metric=metric,
                predictions=predictions,
                references=references,
                perform_validations_in_apply_metric=False,
            )
        self.assertEqual(
            str(cm.exception),
            "Expecting a list of references for each prediction in F1Micro metric. Received reference of type <class 'str'>: dog",
        )

        references = [["cat", "dog"], ["dog"]]
        predictions = ["cat", "dog"]
        with self.assertRaises(ValueError) as cm:
            apply_metric(metric=metric, predictions=predictions, references=references)
        self.assertEqual(
            str(cm.exception),
            "Expecting a list with a single reference per prediction in F1Micro metric. Received a list with multiple references: ['cat', 'dog']",
        )
        references = [[["cat", "dog"]], ["dog"]]
        predictions = ["cat", "dog"]
        with self.assertRaises(ValueError) as cm:
            apply_metric(metric=metric, predictions=predictions, references=references)
        self.assertEqual(
            str(cm.exception),
            "Each reference is expected to be of type 'str' in F1Micro metric. Received reference of type <class 'list'>: ['cat', 'dog']",
        )
        references = [["cat"], ["dog"]]
        predictions = [["cat", "dog"], "dog"]
        with self.assertRaises(ValueError) as cm:
            apply_metric(metric=metric, predictions=predictions, references=references)
        self.assertEqual(
            str(cm.exception),
            "Each prediction is expected to be of type 'str' in F1Micro metric. Received prediction of type <class 'list'>: ['cat', 'dog']",
        )

    def test_f1_binary(self):
        metric = F1Binary()
        references = [[1], [0], [0], [0], [1], [1]]
        predictions = [0.8, 1, 0.2, 0, 0.6, 1]

        global_target = 0.8571428571428
        global_target_neg = 0.8
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )

        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        self.assertAlmostEqual(
            global_target_neg, outputs[0]["score"]["global"]["f1_binary_neg"]
        )
        self.assertEqual("f1_binary", outputs[0]["score"]["global"]["score_name"])
        self.assertEqual("f1_binary", outputs[0]["score"]["instance"]["score_name"])

        metric_pos = F1BinaryPosOnly()
        outputs = apply_metric(
            metric=metric_pos, predictions=predictions, references=references
        )
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        self.assertIsNone(outputs[0]["score"]["global"].get("f1_binary_neg"))

    def test_precision_binary(self):
        metric = PrecisionBinary()
        references = [[1], [0], [0], [0.0], [1.0], [1]]
        predictions = [0.9, 0.6, 0, 0.2, 1, 0.8]

        global_target = 0.75
        global_target_neg = 1
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )

        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        self.assertAlmostEqual(
            global_target_neg, outputs[0]["score"]["global"]["precision_binary_neg"]
        )
        self.assertEqual(
            "precision_binary", outputs[0]["score"]["global"]["score_name"]
        )
        self.assertEqual(
            "precision_binary", outputs[0]["score"]["instance"]["score_name"]
        )

    def test_recall_binary(self):
        metric = RecallBinary()
        references = [[1], [0], [0], [0], [1], [1]]
        predictions = [0.9, 0.6, 0, 0.2, 1, 0.8]

        global_target = 1
        global_target_neg = 0.666666666
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )

        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        self.assertAlmostEqual(
            global_target_neg, outputs[0]["score"]["global"]["recall_binary_neg"]
        )
        self.assertEqual("recall_binary", outputs[0]["score"]["global"]["score_name"])
        self.assertEqual("recall_binary", outputs[0]["score"]["instance"]["score_name"])

    def test_max_f1(self):
        metric = BinaryMaxF1()
        references = [[1], [0], [0], [0]]
        predictions = [0.3, 0, 0.7, 0]

        global_target = 0.666666666666
        global_target_neg = 0.8
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )

        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        self.assertAlmostEqual(
            global_target_neg, outputs[0]["score"]["global"]["max_f1_binary_neg"]
        )
        self.assertEqual("max_f1_binary", outputs[0]["score"]["global"]["score_name"])
        self.assertEqual("max_f1_binary", outputs[0]["score"]["instance"]["score_name"])

    def test_max_f1_single_class(self):
        metric = BinaryMaxF1()
        references = [[0], [0], [0], [0]]
        predictions = [0.3, 0, 0.7, 0]

        global_target = 0.0
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])

    def test_accuracy_binary(self):
        metric = BinaryAccuracy()
        references = [[1], [0], [0], [1], [0]]
        predictions = [0.3, 0, 0.7, 1.0, 0.2]

        expected_global_result = {
            "accuracy_binary": 3 / 5,
            "score": 3 / 5,
            "score_name": "accuracy_binary",
        }

        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        global_result = {
            k: v
            for k, v in outputs[0]["score"]["global"].items()
            if k in expected_global_result
        }
        self.assertDictEqual(expected_global_result, global_result)

    def test_binary_max_accuracy(self):
        metric = BinaryMaxAccuracy()
        references = [[1], [0], [0], [1], [0]]
        predictions = [0.3, 0, 0.7, 1.0, 0.2]

        global_target = 0.8
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )

        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        self.assertEqual(
            "max_accuracy_binary", outputs[0]["score"]["global"]["score_name"]
        )
        self.assertEqual(
            "max_accuracy_binary", outputs[0]["score"]["instance"]["score_name"]
        )

        references = [[0], [0], [0]]
        predictions = [0.3, 0.9, 0.7]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(1.0, outputs[0]["score"]["global"]["score"])

        references = [[1], [0], [0], [1], [0], [0]]
        predictions = [0.7, 0.3, 0.7, 0.8, 0.9, 0.3]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(2 / 3, outputs[0]["score"]["global"]["score"])

        references = [[1]]
        predictions = [0.7]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(1.0, outputs[0]["score"]["global"]["score"])

        references = [[0]]
        predictions = [0.7]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(1.0, outputs[0]["score"]["global"]["score"])

        references = [[0]]
        predictions = [1.7]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(1.0, outputs[0]["score"]["global"]["score"])

    def test_f1_macro(self):
        metric = F1Macro()
        references = [["cat"], ["dog"], ["dog"], ["dog"], ["cat"], ["cat"]]
        predictions = ["cat", "cat", "dog", "dog", "cat", "cat"]
        # recall class 'dog'  = 2/3  = 0.666        precision= 2/2 = 1    f1 = 0.8
        # recall class 'cat'  = 3/3  = 1            precision= 3/4 = 0.75 f1 = 0.857142857143
        # macro f1 = (0.8+0.847)/2
        global_target = 0.82857142
        global_target_dog = 0.8
        global_target_cat = 0.857142857143

        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        self.assertAlmostEqual(
            global_target_dog, outputs[0]["score"]["global"]["f1_dog"]
        )
        self.assertAlmostEqual(
            global_target_cat, outputs[0]["score"]["global"]["f1_cat"]
        )
        self.assertEqual("f1_macro", outputs[0]["score"]["global"]["score_name"])
        self.assertEqual("f1_macro", outputs[0]["score"]["instance"]["score_name"])

    def test_f1_macro_fast(self):
        metric = F1Fast(
            main_score="f1_macro",
            averages=["macro", "per_class"],
            ci_score_names=["f1_macro"],
        )
        references = [["cat"], ["dog"], ["dog"], ["dog"], ["cat"], ["cat"]]
        predictions = ["cat", "cat", "dog", "dog", "cat", "cat"]

        # recall class 'dog'  = 2/3  = 0.666        precision= 2/2 = 1    f1 = 0.8
        # recall class 'cat'  = 3/3  = 1            precision= 3/4 = 0.75 f1 = 0.857142857143
        # macro f1 = (0.8+0.847)/2
        global_target = 0.82857142
        global_target_dog = 0.8
        global_target_cat = 0.857142857143

        outputs = apply_metric(
            metric=metric,
            predictions=predictions,
            references=references,
        )
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        self.assertAlmostEqual(
            global_target_dog, outputs[0]["score"]["global"]["f1_dog"]
        )
        self.assertAlmostEqual(
            global_target_cat, outputs[0]["score"]["global"]["f1_cat"]
        )
        self.assertEqual("f1_macro", outputs[0]["score"]["global"]["score_name"])
        self.assertEqual("f1_macro", outputs[0]["score"]["instance"]["score_name"])

    def test_f1_weighted(self):
        metric = F1Weighted()
        references = [
            ["cat"],
            ["dog"],
            ["dog"],
            ["dog"],
            ["cat"],
            ["cat"],
            ["dog"],
            ["dog"],
        ]
        predictions = ["cat", "cat", "dog", "cat", "cat", "cat", "cat", "dog"]
        # recall class 'dog'  = 2/5  = 0.4          precision= 2/2 = 1    f1 = 0.66666666
        # recall class 'cat'  = 3/3  = 1            precision= 3/6 = 0.5  f1 = 0.57142857
        # weighted f1 = (0.375 * 0.66666666) + (0.625 * 0.57142857) = 0.60714285
        global_target = 0.60714285

        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        self.assertEqual("f1_weighted", outputs[0]["score"]["global"]["score_name"])
        self.assertEqual("f1_weighted", outputs[0]["score"]["instance"]["score_name"])

    def test_f1_macro_with_ood_predictions(self):
        metric = F1Macro()
        references = [["cat"], ["dog"], ["dog"], ["dog"], ["cat"], ["cat"]]
        predictions = ["cat", "2", "dog", "dog", "cat", "cat"]
        # recall class 'dog'  = 2/3  = 0.666        precision= 2/2 = 1    f1 = 0.8
        # recall class 'cat'  = 3/3  = 1            precision= 3/3 = 1    f1  =1
        # macro f1 = 0.9
        global_target = 0.9
        global_target_dog = 0.8
        global_target_cat = 1

        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(
            global_target_dog, outputs[0]["score"]["global"]["f1_dog"]
        )
        self.assertAlmostEqual(
            global_target_cat, outputs[0]["score"]["global"]["f1_cat"]
        )
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        self.assertEqual("f1_macro", outputs[0]["score"]["global"]["score_name"])
        self.assertEqual("f1_macro", outputs[0]["score"]["instance"]["score_name"])

    def test_f1_macro_multilabel(self):
        metric = F1MacroMultiLabel()
        references = [
            [["cat", "dog"]],
            [["dog"]],
            [["dog"]],
            [["dog"]],
            [["cat"]],
            [["cat"]],
        ]
        predictions = [["cat"], ["2"], ["cat", "dog"], ["dog"], ["cat"], ["cat"]]
        # recall class 'dog'  = 2/4  = 0.5          precision= 2/2 = 1       f1 = 0.666666666667
        # recall class 'cat'  = 3/3  = 1            precision= 3/4 = 0.75    f1 = 0.857142857143
        # macro f1 = 0.9
        global_target = 0.76190476
        global_target_dog = 0.666666666667
        global_target_cat = 0.857142857143

        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(
            global_target_dog, outputs[0]["score"]["global"]["f1_dog"]
        )
        self.assertAlmostEqual(
            global_target_cat, outputs[0]["score"]["global"]["f1_cat"]
        )
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        self.assertEqual("f1_macro", outputs[0]["score"]["global"]["score_name"])
        self.assertEqual("f1_macro", outputs[0]["score"]["instance"]["score_name"])

    def test_f1_micro_multilabel(self):
        metric = F1MicroMultiLabel()
        references = [
            [["cat", "dog"]],
            [["dog"]],
            [["dog"]],
            [["dog"]],
            [["cat"]],
            [["cat"]],
        ]
        predictions = [["cat"], ["2"], ["cat", "dog"], ["dog"], ["cat"], ["cat"]]
        # cat     TP=3  FP=1  FN=0  TN=2
        # dog     TP=2  FP=0  FN=2  TN=2
        # total   TP=5  FP=1  FN=2  TN=4
        # precision = TP / (FP + TP) = 5 / 6 = 0.8333333333
        # recall = TP /( FN + TP) =  5 / 7 = 0.7142857
        global_target = 0.769230760933

        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])

    def test_f1_micro_multilabel_error_format(self):
        metric = F1MicroMultiLabel()
        references = [["A B"], ["BC D"], ["C"], ["123"]]
        predictions = [
            ["B", "AB", "A"],
            ["A", "bC", "BC DF"],
            ["c", " C"],
            [13, 23, 234],
        ]
        with self.assertRaises(Exception) as cm:
            apply_metric(metric=metric, predictions=predictions, references=references)

        self.assertEqual(
            str(cm.exception),
            "Each reference is expected to be of type 'List[str]' in F1MicroMultiLabel metric. Received reference of type <class 'str'>: A B",
        )

        references2 = [["A", "B"], ["BC", "D"], ["C"], ["123"]]

        with self.assertRaises(Exception) as cm:
            apply_metric(metric=metric, predictions=predictions, references=references2)

        self.assertEqual(
            str(cm.exception),
            "Expecting a list with a single reference per prediction in F1MicroMultiLabel metric. Received a list with multiple references: ['A', 'B']",
        )

        references3 = [[["A"]], [["BC"]], [["C"]], [["123"]]]  # OK references

        with self.assertRaises(Exception) as cm:
            apply_metric(metric=metric, predictions=predictions, references=references3)

        self.assertEqual(
            str(cm.exception),
            "Each prediction is expected to be of type 'List[str]' in F1MicroMultiLabel metric. Received prediction of type <class 'list'>: [13, 23, 234]",
        )

    def test_f1_macro_multilabel_with_nones(self):
        metric = F1MacroMultiLabel()

        references = [[[]]]
        predictions = [[]]
        global_target = float("nan")
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertTrue(isnan(outputs[0]["score"]["global"]["score"]))

        references = [[[]]]
        predictions = [["x", "y"]]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertTrue(isnan(outputs[0]["score"]["global"]["score"]))

        references = [[["x"]], [["y"]]]
        predictions = [["x"], ["x"]]
        global_target = 0.33333333333
        # Recall(x) = 1.0 Precion(x) = 0.5   --> F1(x) = 0.66666
        # recall(y) = 0.0 Precision(x) = NAN --> F1(y) = 0
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])

        references = [[[]], [["x"]], [["y"]], [[]], [[]]]
        predictions = [[], ["x"], ["x"], [], []]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])

    def test_f1_micro_multilabel_with_nones(self):
        metric = F1MicroMultiLabel()
        references = [[[]]]
        predictions = [["cat", "dog"]]

        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertTrue(isnan(outputs[0]["score"]["global"]["score"]))

        references = [[[]]]
        predictions = [[]]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertTrue(isnan(outputs[0]["score"]["global"]["score"]))

        references = [[["sad"]], [["sad"]]]
        predictions = [["dog", "fustrated"], ["sad"]]
        # TP = 1 FN = 1 FP=0 .. precision=100 recall=0.5
        # sad  TP=1  FP=1  FN=0  TN=1
        #
        # precision = TP / (FP + TP) = 1 / 2 = 0.5
        # recall = TP /( FN + TP) =  1 / 1 = 1

        global_target = 0.66666666
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])

        references = [[[]], [["sad"]]]
        predictions = [["dog", "fustrated"], ["sad"]]
        # precision = TP / (FP + TP) = 1 / 1 = 1
        # recall = TP /( FN + TP) =  1 / 1 = 1

        global_target = 1
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])

    def test_f1_multiple_use(self):
        metric = F1MacroMultiLabel()
        references = [[["cat", "dog"]]]
        predictions = [["cat"]]
        # recall class 'dog'  = 0/1 = 0             precision= 0/0 = 1       f1 = 0
        # recall class 'cat'  = 1/1 = 1             precision= 1/1 = 1       f1 = 1
        global_target = 0.5
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        references = [[["horse"]]]
        predictions = [["horse"]]
        global_target = 1
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])

    def test_key_value_extraction(self):
        metric = KeyValueExtraction(metric="metrics.accuracy")
        # key1 - 2 correct of 2
        # key2 - 1 correct of 2
        # key3 - 0 correct of 1
        # legal keys - 4 out of 5
        references = [ [{"key1": "value1" , "key2" :  "values2"    , "key3": "value3"}], [{"key1": "value3" , "key2" :  "value4"}]]
        predictions = [ {"key1": "value1" , "key2" :  "wrong-value", "wrong-key" : "values3" },{"key1": "value3",  "key2" : "value4", "key3" : "value9"}]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual((2+1+0)/(2 + 2 + 2), outputs[0]["score"]["global"]["accuracy_micro"])
        self.assertAlmostEqual((2/2 + 1/2 + 0/2)/3, outputs[0]["score"]["global"]["accuracy_macro"])
        self.assertAlmostEqual(2/2, outputs[0]["score"]["global"]["accuracy_key1"])
        self.assertAlmostEqual(1/2, outputs[0]["score"]["global"]["accuracy_key2"])
        self.assertAlmostEqual(0/2, outputs[0]["score"]["global"]["accuracy_key3"])
        self.assertAlmostEqual(5/6, outputs[0]["score"]["global"]["accuracy_legal_keys_in_predictions"])


        references = [ [{"key1": "value1" , "key2" :  "values2"    , "key3": "value3"}] ]
        predictions = [ {} ]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(0, outputs[0]["score"]["global"]["accuracy_legal_keys_in_predictions"])

    def test_key_value_extraction_token_overlap(self):
        metric = KeyValueExtraction(metric="metrics.token_overlap",score_prefix="token_overlap_")
        # key1 - recall 1/2, precision 1 , f1 = 2/3
        # key2 - recall 1, precision 0 , f1 = 1
        # legal keys - 2 out of 3
        references = [ [{"address": "IBM" , "zip" :  "32312"} ] ]
        predictions = [ {"address": "IBM Corp", "zip" : "32312", "user" : "george"} ]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(2/3, outputs[0]["score"]["global"]["token_overlap_f1_address"])
        self.assertAlmostEqual(1, outputs[0]["score"]["global"]["token_overlap_f1_zip"])
        self.assertAlmostEqual(2/3, outputs[0]["score"]["global"]["token_overlap_f1_legal_keys_in_predictions"])
        self.assertAlmostEqual((2/3 + 1)/2, outputs[0]["score"]["global"]["token_overlap_f1_micro"])
        self.assertAlmostEqual((2/3 + 1)/2, outputs[0]["score"]["global"]["token_overlap_f1_macro"])




    def test_rouge(self):
        metric = Rouge()
        references = [["hello", "there"], ["general kenobi", "general yoda"]]
        predictions = ["hello there", "general kenobi"]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        global_target = 5 / 6
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])

        # compare with the HF implementation
        class OldRouge(HuggingfaceMetric):
            hf_metric_name = "rouge"
            main_score = "rougeL"
            scale = 1.0

            prediction_type = "str"
            single_reference_per_prediction = False  # multiple references allowed

            use_aggregator: bool = True
            rouge_types: List[str] = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

            sent_split_newline: bool = True

            _requirements_list: List[str] = ["nltk", "rouge_score"]

            def prepare(self):
                super().prepare()

                self.hf_compute_args.update(
                    {
                        "use_aggregator": self.use_aggregator,
                        "rouge_types": self.rouge_types,
                    }
                )

                import nltk

                nltk.download("punkt_tab", quiet=True)
                self.sent_tokenize = nltk.sent_tokenize

            def compute(self, references, predictions, task_data: List[Dict]):
                if self.sent_split_newline:
                    predictions = [
                        "\n".join(self.sent_tokenize(prediction.strip()))
                        for prediction in predictions
                    ]
                    references = [
                        ["\n".join(self.sent_tokenize(r.strip())) for r in reference]
                        for reference in references
                    ]
                return super().compute(references, predictions, task_data)

        metric = OldRouge()
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])

    def test_token_overlap(self):
        metric = TokenOverlap()
        predictions = ["hello there general dude", "foo bar foobar"]
        references = [
            ["hello there general kenobi", "hello there!"],
            ["foo bar foobar", "foo bar"],
        ]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        global_targets = {"f1": 7 / 8, "precision": 7 / 8, "recall": 1}
        for target, value in global_targets.items():
            self.assertAlmostEqual(value, outputs[0]["score"]["global"][target])

    def test_roc_auc(self):
        metric = RocAuc()
        predictions = [0.2, 0.8, 1.0]
        references = [[1.0], [0.0], [1.0]]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        global_target = 0.5
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])

    def test_kendalltau(self):
        metric = KendallTauMetric()
        predictions = [1.0, 2.0, 1.0]
        references = [[-1.0], [1.0], [0.0]]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        global_target = 0.81649658092772
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])

    def test_detector(self):
        metric = Detector(model_name="MilaNLProc/bert-base-uncased-ear-misogyny")
        predictions = ["I hate women.", "I do not hate women."]
        references = [["I hate women."], ["I do not hate women."]]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        global_target = 0.9562818706035614
        self.assertAlmostEqual(
            global_target, outputs[0]["score"]["global"]["score"], places=4
        )

    def test_normalized_sacrebleu(self):
        metric = NormalizedSacrebleu()
        predictions = ["hello there general kenobi", "foo bar foobar"]
        references = [
            ["hello there general kenobi", "hello there !"],
            ["foo bar foobar", "foo bar foobar"],
        ]
        task_data = [{"tokenize": None}, {"tokenize": None}]

        outputs = apply_metric(
            metric=metric,
            predictions=predictions,
            references=references,
            task_data=task_data,
        )
        global_target = 1.0
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])

    def test_ner(self):
        metric = NER()
        predictions = [
            [
                ("Dalia", "Person"),
                ("Ramat-Gan", "Location"),
                ("IBM", "Org"),
            ]
        ]
        references = [
            [
                [
                    ("Dalia", "Person"),
                    ("Givataaim", "Location"),
                ]
            ]
        ]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        global_target = 1.0
        self.assertAlmostEqual(
            global_target, outputs[0]["score"]["global"]["f1_Person"]
        )
        global_target = 0.0
        self.assertAlmostEqual(
            global_target, outputs[0]["score"]["global"]["f1_Location"]
        )
        metric.report_per_group_scores = False
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertTrue("f1_Person" not in outputs[0]["score"]["global"])
        self.assertTrue("f1_Location" not in outputs[0]["score"]["global"])

    def test_llama_index_correctness(self):
        metric = LlamaIndexCorrectness(model_name="mock")
        predictions = ["1976"]
        references = [["1976"]]
        task_data = [
            {
                "group_id": "group1",
                "variant_type": "original",
                "question": "what year is it",
                "contexts": ["the year is 1976"],
            },
        ]

        instance_targets = [
            {
                "correctness_llama_index_by_mock_judge": 1.0,
                "score": 1.0,
                "score_name": "correctness_llama_index_by_mock_judge",
            }
        ]
        global_target = 1.0
        outputs = apply_metric(
            metric=metric,
            predictions=predictions,
            references=references,
            task_data=task_data,
        )

        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        for output, target in zip(outputs, instance_targets):
            self.assertEqual(output["score"]["instance"], target)

    def test_grouped_instance_metrics(self):
        accuracy_metrics = [
            FixedGroupMeanAccuracy(),
            GroupMeanAccuracy(),
            FixedGroupMeanStringContainment(),
            GroupMeanStringContainment(),
            FixedGroupMeanBaselineAccuracy(),
            FixedGroupMeanParaphraseAccuracy(),
            FixedGroupMeanBaselineStringContainment(),
            FixedGroupMeanParaphraseStringContainment(),
            GroupMeanTokenOverlap(),
            FixedGroupNormCohensHParaphraseAccuracy(),
            FixedGroupNormCohensHParaphraseStringContainment(),
            FixedGroupPDRParaphraseAccuracy(),
            FixedGroupPDRParaphraseStringContainment(),
            FixedGroupNormHedgesGParaphraseAccuracy(),
            FixedGroupNormHedgesGParaphraseStringContainment(),
            FixedGroupAbsvalNormCohensHParaphraseAccuracy(),
            FixedGroupAbsvalNormCohensHParaphraseStringContainment(),
            FixedGroupAbsvalNormHedgesGParaphraseAccuracy(),
            FixedGroupAbsvalNormHedgesGParaphraseStringContainment(),
        ]
        global_targets = [
            0.225,
            0.225,
            0.4875,
            0.4875,
            0.5,
            0.19444444444444442,
            0.75,
            0.5555555555555555,
            0.5083333333333333,
            -0.4249467048786864,
            -0.4639421840102023,
            0.8333333333333334,
            0.4444444444444445,
            -0.34565986391520215,
            -0.08060156608173413,
            0.6471689271009087,
            0.4639421840102023,
            0.3832160660602437,
            0.08060156608173413,
        ]

        for metric, target in zip(accuracy_metrics, global_targets):
            for score_prefix in ["my_", ""]:
                metric.score_prefix = score_prefix
                outputs = apply_metric(
                    metric=metric,
                    predictions=GROUPED_INSTANCE_PREDICTIONS,
                    references=GROUPED_INSTANCE_REFERENCES,
                    task_data=GROUPED_INSTANCE_ADDL_INPUTS,
                )

                self.assertAlmostEqual(
                    target,
                    outputs[0]["score"]["global"]["score"],
                    msg=f"metric {metric.__class__.__name__} output {outputs[0]['score']['global']['score_name']} does not equal the expected value {target}",
                )
                self.assertEqual(
                    outputs[0]["score"]["global"]["num_of_instances"],
                    len(GROUPED_INSTANCE_ADDL_INPUTS),
                )

                end_of_main_score_name_in_global = "_".join(
                    [
                        metric.reduction_map["group_mean"]["agg_func"][0],
                        score_prefix,
                        metric.main_score,
                    ]
                ).replace(
                    "__", "_"
                )  # for the case of empty score_prefix

                self.assertTrue(
                    any(
                        key.endswith(end_of_main_score_name_in_global)
                        for key in outputs[0]["score"]["global"]
                    )
                )

    def test_grouped_instance_metric_errors(self):
        """Test certain value and assertion error raises for grouped instance metrics (with group_mean reduction)."""
        from dataclasses import field
        from statistics import mean
        from typing import List

        class NoAggFuncReduction(Accuracy):
            implemented_reductions: List[str] = field(
                default_factory=lambda: ["mean", "group_mean", "some_other_func"]
            )
            reduction_map = {"some_other_func": {"agg_func": ["mean", mean, False]}}

        with self.assertRaises(ValueError):
            # should raise error because no aggregation_function will be defined, since only mean and group_mean are implemented
            metric = NoAggFuncReduction()
            apply_metric(
                metric=metric,
                predictions=GROUPED_INSTANCE_PREDICTIONS,
                references=GROUPED_INSTANCE_REFERENCES,
                task_data=GROUPED_INSTANCE_ADDL_INPUTS,
            )

        class NoAggFunc(Accuracy):
            reduction_map = {"group_mean": {"func": ["mean", mean]}}

        with self.assertRaises(AssertionError):
            # should raise error because no "agg_func" field in group_mean
            metric = NoAggFunc()
            apply_metric(
                metric=metric,
                predictions=GROUPED_INSTANCE_PREDICTIONS,
                references=GROUPED_INSTANCE_REFERENCES,
                task_data=GROUPED_INSTANCE_ADDL_INPUTS,
            )

        class NoCallableAggFunc(Accuracy):
            reduction_map = {"group_mean": {"agg_func": ["mean", "some string", False]}}

        with self.assertRaises(AssertionError):
            # should raise error because second field of agg_func should be callable
            metric = NoCallableAggFunc()
            apply_metric(
                metric=metric,
                predictions=GROUPED_INSTANCE_PREDICTIONS,
                references=GROUPED_INSTANCE_REFERENCES,
                task_data=GROUPED_INSTANCE_ADDL_INPUTS,
            )

        class NoBooleanGrouping(Accuracy):
            reduction_map = {"group_mean": {"agg_func": ["mean", mean, 1]}}

        with self.assertRaises(AssertionError):
            # should raise error because third field in agg_func is not boolean
            metric = NoBooleanGrouping()
            apply_metric(
                metric=metric,
                predictions=GROUPED_INSTANCE_PREDICTIONS,
                references=GROUPED_INSTANCE_REFERENCES,
                task_data=GROUPED_INSTANCE_ADDL_INPUTS,
            )

    def test_run_metric_with_different_fields(self):
        metric = Accuracy(reference_field="my_field")
        outputs = apply_metric(
            metric=metric,
            predictions=["A"],
            references=[["B"]],
            task_data=[{"my_field": "A"}],
        )
        target = 1.0
        self.assertEqual(outputs[0]["score"]["global"]["score"], target)

        metric = Accuracy(prediction_field="my_field")
        outputs = apply_metric(
            metric=metric,
            predictions=["A"],
            references=[["B"]],
            task_data=[{"my_field": "B"}],
        )
        target = 1.0
        self.assertEqual(outputs[0]["score"]["global"]["score"], target)

    def test_tool_calling_metric(self):
        tools_data = {
            "__tools__": [{
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "parameters": {
                        "properties": {
                            "param1": {"type": "string"},
                            "param2": {"type": "integer"}
                        }
                    }
                }
            }]
        }

        # Test case 1: Exact match
        prediction = {"name": "test_tool", "arguments": {"param1": "value1"}}
        reference = {"name": "test_tool", "arguments": {"param1": "value1"}}

        outputs = apply_metric(
            metric=ToolCallingMetric(),
            predictions=[prediction],
            references=[[reference]],
            task_data=[tools_data]
        )

        # Exact match should be 1.0 when prediction and reference are identical
        self.assertEqual(outputs[0]["score"]["global"]["exact_match"], 1.0)
        self.assertEqual(outputs[0]["score"]["global"]["tool_choice"], 1.0)
        self.assertEqual(outputs[0]["score"]["global"]["parameter_choice"], 1.0)
        self.assertEqual(outputs[0]["score"]["global"]["parameter_values"], 1.0)

        # Test case 2: Different tool name
        prediction = {"name": "different_tool", "arguments": {"param1": "value1"}}
        reference = {"name": "test_tool", "arguments": {"param1": "value1"}}

        outputs = apply_metric(
            metric=ToolCallingMetric(),
            predictions=[prediction],
            references=[[reference]],
            task_data=[tools_data]
        )

        # Exact match and tool choice should be 0.0, but parameter choice can still match
        self.assertEqual(outputs[0]["score"]["global"]["exact_match"], 0.0)
        self.assertEqual(outputs[0]["score"]["global"]["tool_choice"], 0.0)
        self.assertEqual(outputs[0]["score"]["global"]["parameter_choice"], 1.0)

        # Test case 3: Different parameter names
        prediction = {"name": "test_tool", "arguments": {"param1": "value1", "wrongParam": "value2"}}
        reference = {"name": "test_tool", "arguments": {"param1": "value1", "param2": 42}}

        outputs = apply_metric(
            metric=ToolCallingMetric(),
            predictions=[prediction],
            references=[[reference]],
            task_data=[tools_data]
        )

        # Exact match should be 0.0, tool choice 1.0, parameter choice 0.5 (half match)
        self.assertEqual(outputs[0]["score"]["global"]["exact_match"], 0.0)
        self.assertEqual(outputs[0]["score"]["global"]["tool_choice"], 1.0)
        self.assertEqual(outputs[0]["score"]["global"]["parameter_choice"], 0.5)

        # Test case 4: Different parameter values
        prediction = {"name": "test_tool", "arguments": {"param1": "different", "param2": 42}}
        reference = {"name": "test_tool", "arguments": {"param1": "value1", "param2": 123}}

        outputs = apply_metric(
            metric=ToolCallingMetric(),
            predictions=[prediction],
            references=[[reference]],
            task_data=[tools_data]
        )

        # Parameter choice should be 1.0 (all names match), but parameter values 0.5 (one match)
        self.assertEqual(outputs[0]["score"]["global"]["tool_choice"], 1.0)
        self.assertEqual(outputs[0]["score"]["global"]["parameter_choice"], 1.0)
        self.assertEqual(outputs[0]["score"]["global"]["parameter_values"], 0.0)

        # Test case 5: Empty arguments
        prediction = {"name": "test_tool", "arguments": {}}
        reference = {"name": "test_tool", "arguments": {"param1": "value1"}}

        outputs = apply_metric(
            metric=ToolCallingMetric(),
            predictions=[prediction],
            references=[[reference]],
            task_data=[tools_data]
        )

        # Parameter choice should be 1.0 for empty arguments
        self.assertEqual(outputs[0]["score"]["global"]["parameter_choice"], 1.0)
        self.assertEqual(outputs[0]["score"]["global"]["parameter_values"], 1.0)

        # Test case 6: Multiple references with one match
        prediction = {"name": "test_tool", "arguments": {"param1": "value1"}}
        references = [
            {"name": "wrong_tool", "arguments": {"param1": "value1"}},
            {"name": "test_tool", "arguments": {"param1": "value1"}}
        ]

        outputs = apply_metric(
            metric=ToolCallingMetric(),
            predictions=[prediction],
            references=[references],
            task_data=[tools_data]
        )

        # Should match the exact reference
        self.assertEqual(outputs[0]["score"]["global"]["exact_match"], 1.0)
        self.assertEqual(outputs[0]["score"]["global"]["tool_choice"], 1.0)

        # Test case 7: Parameter types
        prediction = {"name": "test_tool", "arguments": {"param1": "string", "param2": 42}}

        outputs = apply_metric(
            metric=ToolCallingMetric(),
            predictions=[prediction],
            references=[[]],  # Empty references
            task_data=[tools_data]
        )

        # Parameters should have correct types
        self.assertEqual(outputs[0]["score"]["global"]["parameters_schema_validation"], 1.0)

        # Test case 8: Wrong parameter types
        prediction = {"name": "test_tool", "arguments": {"param1": "string", "param2": "not_an_integer"}}

        outputs = apply_metric(
            metric=ToolCallingMetric(),
            predictions=[prediction],
            references=[[]],
            task_data=[tools_data]
        )

        # Only half of parameters have correct types
        self.assertEqual(outputs[0]["score"]["global"]["parameters_schema_validation"], 0.0)

    def test_perplexity(self):
        prediction = ["who are we?"]
        references = [["we are the world"]]

        perplexity_question = Perplexity(
            model_name="google/flan-t5-small",
            source_template="Generate a question based on the given content: {reference}",
            target_template="{prediction}",
        )
        first_instance_target = 0.059865921735763
        outputs = apply_metric(
            metric=perplexity_question, predictions=prediction, references=references
        )
        self.assertAlmostEqual(
            first_instance_target, outputs[0]["score"]["instance"]["score"]
        )

    def test_fuzzyner(self):
        metric = FuzzyNer()
        predictions = [
            [
                ("jar htaras", "Person"),
                ("Marathahalli", "Location"),
                ("IBM", "Org"),
            ]
        ]
        references = [
            [
                [
                    ("jar htaras", "Person"),
                    ("Marathahalli ring road", "Location"),
                ]
            ]
        ]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        global_target = 1.0
        self.assertAlmostEqual(
            global_target, outputs[0]["score"]["global"]["f1_Person"]
        )
        global_target = 0.0
        self.assertAlmostEqual(
            global_target, outputs[0]["score"]["global"]["f1_Location"]
        )
        metric.report_per_group_scores = False
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertTrue("f1_Person" not in outputs[0]["score"]["global"])
        self.assertTrue("f1_Location" not in outputs[0]["score"]["global"])

    def test_perplexity_with_prefix(self):
        prediction = ["who are we?"]
        references = [["we are the world"]]

        perplexity_question = Perplexity(
            model_name="google/flan-t5-small",
            source_template="Generate a question based on the given content: {reference}",
            target_template="{prediction}",
            score_prefix="my_",
        )

        outputs = apply_metric(
            metric=perplexity_question, predictions=prediction, references=references
        )

        expected_global_result = {
            "my_perplexity": 0.06,
            "score": 0.06,
            "score_name": "my_perplexity",
            "num_of_instances": 1,
        }

        global_result = outputs[0]["score"]["global"].copy()
        # Only check the keys that are expected, i.e. exist in expected_global_result
        global_result = {
            key: value
            for key, value in global_result.items()
            if key in expected_global_result
        }

        expected_instance_results = [
            {
                "my_perplexity": 0.06,
                "score": 0.06,
                "score_name": "my_perplexity",
                "my_reference_scores": [0.06],
            }
        ]
        check_scores(
            expected_global_result,
            expected_instance_results,
            global_outputs=outputs[0]["score"]["global"],
            instance_outputs=[outputs[0]["score"]["instance"]],
        )

    def test_text2sql_accuracy_correct_query_mock_db(self):
        sql_execution_metric = SQLExecutionAccuracy()
        sql_non_execution_metric = SQLNonExecutionAccuracy()
        predictions = ["SELECT name FROM employees WHERE department = 'Sales'"]
        references = ["SELECT name FROM employees WHERE department = 'Sales';"]
        task_data = [
            {
                "db": {
                    "db_id": "mock_db",
                    "db_type": "in_memory",
                    "data": {
                        "employees": {
                            "columns": ["id", "name", "department", "salary"],
                            "rows": [
                                (1, "Alice", "Sales", 50000),
                                (2, "Bob", "Engineering", 60000),
                                (3, "Charlie", "Sales", 55000),
                            ],
                        }
                    },
                }
            }
        ]

        execution_outputs = sql_execution_metric.compute(
            references, predictions[0], task_data[0]
        )
        self.assertEqual(1.0, execution_outputs["score"])
        non_execution_outputs = sql_non_execution_metric.compute(
            references, predictions[0], task_data[0]
        )
        self.assertEqual(1.0, non_execution_outputs["score"])

    def test_text2sql_accuracy_different_db_schema(self):
        sql_execution_metric = SQLExecutionAccuracy()
        sql_non_execution_metric = SQLNonExecutionAccuracy()
        predictions = [
            "SELECT product_name, price FROM products WHERE category = 'Electronics'"
        ]
        references = [
            "SELECT product_name AS pname, price AS cost FROM products WHERE category = 'Electronics';"
        ]
        task_data = [
            {
                "db": {
                    "db_id": "products_db",
                    "db_type": "in_memory",
                    "data": {
                        "products": {
                            "columns": [
                                "product_id",
                                "product_name",
                                "category",
                                "price",
                            ],
                            "rows": [
                                (1, "Laptop", "Electronics", 1200),
                                (2, "Mouse", "Electronics", 25),
                                (3, "Shirt", "Clothing", 50),
                                (4, "Monitor", "Electronics", 300),
                            ],
                        }
                    },
                }
            }
        ]

        execution_outputs = sql_execution_metric.compute(
            references, predictions[0], task_data[0]
        )
        self.assertEqual(1.0, execution_outputs["score"])
        non_execution_outputs = sql_non_execution_metric.compute(
            references, predictions[0], task_data[0]
        )
        self.assertEqual(1.0, non_execution_outputs["score"])

    def test_text2sql_accuracy_multiple_tables(self):
        sql_execution_metric = SQLExecutionAccuracy()
        sql_non_execution_metric = SQLNonExecutionAccuracy()
        predictions = [
            "SELECT o.order_id, c.name FROM orders AS o JOIN customers AS c ON o.customer_id = c.customer_id WHERE o.status = 'Shipped'"
        ]
        references = [
            "SELECT o.order_id, c.name FROM orders AS o INNER JOIN customers AS c ON o.customer_id = c.customer_id WHERE o.status = 'Shipped';"
        ]
        task_data = [
            {
                "db": {
                    "db_id": "sales_db",
                    "db_type": "in_memory",
                    "data": {
                        "customers": {
                            "columns": ["customer_id", "name", "city"],
                            "rows": [
                                (1, "John Doe", "New York"),
                                (2, "Jane Smith", "Los Angeles"),
                                (3, "David Lee", "Chicago"),
                            ],
                        },
                        "orders": {
                            "columns": ["order_id", "customer_id", "status"],
                            "rows": [
                                (101, 1, "Shipped"),
                                (102, 2, "Pending"),
                                (103, 1, "Shipped"),
                            ],
                        },
                    },
                }
            }
        ]

        execution_outputs = sql_execution_metric.compute(
            references, predictions[0], task_data[0]
        )
        self.assertEqual(1.0, execution_outputs["score"])
        non_execution_outputs = sql_non_execution_metric.compute(
            references, predictions[0], task_data[0]
        )
        self.assertEqual(1.0, non_execution_outputs["score"])

    def test_text2sql_accuracy_empty_result(self):
        sql_execution_metric = SQLExecutionAccuracy()
        sql_non_execution_metric = SQLNonExecutionAccuracy()
        predictions = ["SELECT name FROM employees WHERE department = 'HR'"]
        references = ["SELECT name FROM employees WHERE department = 'HR';"]
        task_data = [
            {
                "db": {
                    "db_id": "mock_db",
                    "db_type": "in_memory",
                    "data": {
                        "employees": {
                            "columns": ["id", "name", "department", "salary"],
                            "rows": [
                                (1, "Alice", "Sales", 50000),
                                (2, "Bob", "Engineering", 60000),
                                (3, "Charlie", "Sales", 55000),
                            ],
                        }
                    },
                }
            }
        ]

        execution_outputs = sql_execution_metric.compute(
            references, predictions[0], task_data[0]
        )
        self.assertEqual(0.0, execution_outputs["score"])
        non_execution_outputs = sql_non_execution_metric.compute(
            references, predictions[0], task_data[0]
        )
        self.assertEqual(1.0, non_execution_outputs["score"])

    def test_text2sql_accuracy_aggregation_query(self):
        sql_execution_metric = SQLExecutionAccuracy()
        sql_non_execution_metric = SQLNonExecutionAccuracy()
        predictions = ["SELECT AVG(salary) FROM employees"]
        references = ["SELECT AVG(salary) FROM employees;"]
        task_data = [
            {
                "db": {
                    "db_id": "mock_db",
                    "db_type": "in_memory",
                    "data": {
                        "employees": {
                            "columns": ["id", "name", "department", "salary"],
                            "rows": [
                                (1, "Alice", "Sales", 50000),
                                (2, "Bob", "Engineering", 60000),
                                (3, "Charlie", "Sales", 55000),
                            ],
                        }
                    },
                }
            }
        ]

        execution_outputs = sql_execution_metric.compute(
            references, predictions[0], task_data[0]
        )
        self.assertEqual(1.0, execution_outputs["score"])
        non_execution_outputs = sql_non_execution_metric.compute(
            references, predictions[0], task_data[0]
        )
        self.assertEqual(1.0, non_execution_outputs["score"])

    def test_text2sql_accuracy_incorrect_query(self):
        sql_execution_metric = SQLExecutionAccuracy()
        sql_non_execution_metric = SQLNonExecutionAccuracy()
        predictions = [
            "SELECT nme FROM employees WHERE department = 'Sales'"
        ]  # Incorrect column name 'nme'
        references = ["SELECT name FROM employees WHERE department = 'Sales';"]
        task_data = [
            {
                "db": {
                    "db_id": "mock_db",
                    "db_type": "in_memory",
                    "data": {
                        "employees": {
                            "columns": ["id", "name", "department", "salary"],
                            "rows": [
                                (1, "Alice", "Sales", 50000),
                                (2, "Bob", "Engineering", 60000),
                                (3, "Charlie", "Sales", 55000),
                            ],
                        }
                    },
                }
            }
        ]

        execution_outputs = sql_execution_metric.compute(
            references, predictions[0], task_data[0]
        )
        self.assertEqual(0.0, execution_outputs["score"])
        non_execution_outputs = sql_non_execution_metric.compute(
            references, predictions[0], task_data[0]
        )
        self.assertEqual(0.0, non_execution_outputs["score"])


class TestConfidenceIntervals(UnitxtTestCase):
    def test_confidence_interval_off(self):
        """Test that when metric.n_resamples is set to None, no confidence intervals are computed."""
        # Test one GlobalMetric and one InstanceMetric
        for metric in [Accuracy(), F1Macro()]:
            metric.disable_confidence_interval_calculation()
            outputs = apply_metric(metric=metric, predictions=["A"], references=[["A"]])

            global_result = outputs[0]["score"]["global"]
            # Check there are no confidence intervals in the result
            for key in global_result:
                self.assertTrue("ci_low" not in key)
                self.assertTrue("ci_high" not in key)

    def test_instance_metric_confidence_interval(self):
        """Test the calculation of confidence intervals for an instance metric (Accuracy is used as an instance of an InstanceMetric)."""
        self._test_confidence_interval(
            metric=Accuracy(),
            expected_ci_low=0.71,
            expected_ci_high=0.87,
        )

    def test_map_reduce_metric_confidence_interval(self):
        """Test the calculation of confidence intervals for an instance metric (Accuracy is used as an instance of an InstanceMetric)."""
        self._test_confidence_interval(
            metric=AccuracyFast(),
            expected_ci_low=0.71,
            expected_ci_high=0.87,
        )

    def test_f1_micro_confidence_interval(self):
        """Test the calculation of confidence intervals for an instance metric (Accuracy is used as an instance of an InstanceMetric)."""
        self._test_confidence_interval(
            metric=F1Micro(n_resamples=1000),
            expected_ci_low=0.83,
            expected_ci_high=0.93,
        )

    def test_f1_micro_fast_confidence_interval(self):
        """Test the calculation of confidence intervals for an instance metric (Accuracy is used as an instance of an InstanceMetric)."""
        self._test_confidence_interval(
            metric=F1Fast(main_score="f1_micro", averages=["micro"]),
            expected_ci_low=0.83,
            expected_ci_high=0.93,
        )

    def test_instance_metric_with_multiple_scores_confidence_interval(self):
        self._test_confidence_interval(
            metric=TokenOverlap(),
            expected_ci_low=0.71,
            expected_ci_high=0.87,
        )

    def test_global_metric_confidence_interval(self):
        """Test the calculation of confidence intervals for global metrics (F1Macro and F1Micro are used as instances of a GlobalMetric)."""
        f1_macro_low, f1_macro_high = 0.8809213119223925, 0.9439681645177271
        self._test_confidence_interval(
            metric=F1Macro(),
            expected_ci_low=f1_macro_low,
            expected_ci_high=f1_macro_high,
        )
        f1_micro_low, f1_micro_high = 0.8439306358381503, 0.9223675337263242
        self._test_confidence_interval(
            metric=F1Micro(),
            expected_ci_low=f1_micro_low,
            expected_ci_high=f1_micro_high,
        )

        # Now reverse the order and check things don't change
        self._test_confidence_interval(
            metric=F1Micro(),
            expected_ci_low=f1_micro_low,
            expected_ci_high=f1_micro_high,
        )
        self._test_confidence_interval(
            metric=F1Macro(),
            expected_ci_low=f1_macro_low,
            expected_ci_high=f1_macro_high,
        )

    def _test_confidence_interval(self, metric, expected_ci_low, expected_ci_high):
        """Test the calculation of confidence intervals for a given metric."""
        predictions = ["A", "B", "C", "D", "E"] * 20  # 100 predictions
        references = [["B"], ["B"], ["C"], ["D"], ["E"]] * 20  # 80% are correct (4/5)

        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )

        expected_global_result = {
            f"{metric.main_score}_ci_low": expected_ci_low,
            f"{metric.main_score}_ci_high": expected_ci_high,
            "score_ci_low": expected_ci_low,
            "score_ci_high": expected_ci_high,
        }

        global_result = outputs[0]["score"]["global"].copy()
        logger.info(global_result)
        for score_name, score_value in global_result.items():
            if score_name in expected_global_result:
                # Verify that the output value is as the expected value
                self.assertAlmostEqual(
                    score_value, expected_global_result[score_name], places=3
                )
            else:
                # An output score that is not expected
                # This is ok if the score_name is not related to confidence intervals
                # Otherwise, there was some confidence interval calculation that was not supposed to occur.
                self.assertTrue(
                    ("ci_low" not in score_name and "ci_high" not in score_name)
                    or score_name not in metric.ci_scores,
                    msg=f"Unexpected confidence interval score '{score_name}'.",
                )

    def test_grouped_instance_metric_confidence_interval(self):
        """Test the calculation of confidence intervals for grouped instance metrics (sub-types of InstanceMetric with group_mean reduction)."""
        self._test_grouped_instance_confidence_interval(
            metric=FixedGroupMeanAccuracy(),
            expected_ci_low=0.1,
            expected_ci_high=0.48178555627359004,
        )

        self._test_grouped_instance_confidence_interval(
            metric=GroupMeanAccuracy(),
            expected_ci_low=0.025,
            expected_ci_high=0.4407250456645065,
        )

        self._test_grouped_instance_confidence_interval(
            metric=FixedGroupMeanStringContainment(),
            expected_ci_low=0.0,
            expected_ci_high=0.675,
        )

        self._test_grouped_instance_confidence_interval(
            metric=GroupMeanStringContainment(),
            expected_ci_low=0.15627449950197503,
            expected_ci_high=0.7080527276705952,
        )

        self._test_grouped_instance_confidence_interval(
            metric=FixedGroupMeanBaselineAccuracy(),
            expected_ci_low=0.0,
            expected_ci_high=1.0,
        )

        self._test_grouped_instance_confidence_interval(
            metric=FixedGroupMeanParaphraseAccuracy(),
            expected_ci_low=0.0,
            expected_ci_high=0.3333333333333333,
        )

        self._test_grouped_instance_confidence_interval(
            metric=FixedGroupMeanBaselineStringContainment(),
            expected_ci_low=0.25,
            expected_ci_high=1.0,
        )

        self._test_grouped_instance_confidence_interval(
            metric=FixedGroupMeanParaphraseStringContainment(),
            expected_ci_low=0.5,
            expected_ci_high=0.6666666666666666,
        )

        self._test_grouped_instance_confidence_interval(
            metric=FixedGroupNormCohensHParaphraseAccuracy(),
            expected_ci_low=-1.0,
            expected_ci_high=0.33333333333333337,
        )

        # note, this metric has an issue where the ci_high on PCs on Travis slightly diverges from the local results
        # hence this test may fail on a PC
        self._test_grouped_instance_confidence_interval(
            metric=FixedGroupNormCohensHParaphraseStringContainment(),
            expected_ci_low=-0.49999999999999994,
            expected_ci_high=-0.39182655203060723,
        )

        self._test_grouped_instance_confidence_interval(
            metric=FixedGroupPDRParaphraseAccuracy(),
            expected_ci_low=0.6666666666666666,
            expected_ci_high=1.0,
        )

        self._test_grouped_instance_confidence_interval(
            metric=FixedGroupPDRParaphraseStringContainment(),
            expected_ci_low=0.3333333333333333,
            expected_ci_high=0.5,
        )

        self._test_grouped_instance_confidence_interval(
            metric=FixedGroupNormHedgesGParaphraseAccuracy(),
            expected_ci_low=-1.0,
            expected_ci_high=0.01892225367237965,
        )

        self._test_grouped_instance_confidence_interval(
            metric=FixedGroupNormHedgesGParaphraseStringContainment(),
            expected_ci_low=-0.09757387538180902,
            expected_ci_high=-0.046656947481584346,
        )

        # absolute value of Hedges' g and Cohen's h
        self._test_grouped_instance_confidence_interval(
            metric=FixedGroupAbsvalNormCohensHParaphraseAccuracy(),
            expected_ci_low=0.33333333333333337,
            expected_ci_high=1.0,
        )

        self._test_grouped_instance_confidence_interval(
            metric=FixedGroupAbsvalNormCohensHParaphraseStringContainment(),
            expected_ci_low=0.39182655203060723,
            expected_ci_high=0.49999999999999994,
        )

        self._test_grouped_instance_confidence_interval(
            metric=FixedGroupAbsvalNormHedgesGParaphraseAccuracy(),
            expected_ci_low=0.05633430321756243,
            expected_ci_high=1.0,
        )

        self._test_grouped_instance_confidence_interval(
            metric=FixedGroupAbsvalNormHedgesGParaphraseStringContainment(),
            expected_ci_low=0.046656947481584346,
            expected_ci_high=0.09757387538180902,
        )

        # pass global dict because there are additional fields other than the main score
        for score_prefix in ["my_", ""]:
            self._test_grouped_instance_confidence_interval(
                metric=GroupMeanTokenOverlap(),
                expected_global_result={
                    f"group_mean_{score_prefix}recall": 0.525,
                    f"group_mean_{score_prefix}f1": 0.5083333333333333,
                    "score": 0.5083333333333333,
                    "score_name": f"group_mean_{score_prefix}f1",
                    f"group_mean_{score_prefix}precision": 0.5,
                    f"group_mean_{score_prefix}recall_ci_low": 0.25,
                    f"group_mean_{score_prefix}recall_ci_high": 0.7083333333333334,
                    f"group_mean_{score_prefix}f1_ci_low": 0.22302503471948287,
                    f"group_mean_{score_prefix}f1_ci_high": 0.6805555555555555,
                    "score_ci_low": 0.22302503471948287,
                    "score_ci_high": 0.6805555555555555,
                    f"group_mean_{score_prefix}precision_ci_low": 0.2095091529536007,
                    f"group_mean_{score_prefix}precision_ci_high": 0.6666666666666666,
                },
                input_score_prefixes=[score_prefix],
            )

    def _test_grouped_instance_confidence_interval(
        self,
        metric,
        expected_ci_low=0.0,
        expected_ci_high=1.0,
        expected_global_result=None,
        input_score_prefixes=None,
    ):
        """Test the calculation of confidence intervals for a given metric with group_mean reduction."""
        input_expected_global_result_is_none = expected_global_result is None
        # to remember between score_prefixes

        for score_prefix in (
            ["my_", ""] if input_score_prefixes is None else input_score_prefixes
        ):
            metric.score_prefix = score_prefix
            outputs = apply_metric(
                metric=metric,
                predictions=GROUPED_INSTANCE_PREDICTIONS,
                references=GROUPED_INSTANCE_REFERENCES,
                task_data=GROUPED_INSTANCE_ADDL_INPUTS,
            )
            # get first element of reduction_map values
            reduction_params = next(iter(metric.reduction_map.values()))
            prefix = "fixed_group" if reduction_params["agg_func"][2] else "group"
            group_score_name = "_".join(
                [
                    prefix,
                    metric.reduction_map["group_mean"]["agg_func"][0],
                    score_prefix,
                    metric.main_score,
                ]
            ).replace(
                "__", "_"
            )  # for the case of empty score_prefix

            if input_expected_global_result_is_none:
                expected_global_result = {
                    f"{group_score_name}_ci_low": expected_ci_low,
                    f"{group_score_name}_ci_high": expected_ci_high,
                    "score_ci_low": expected_ci_low,
                    "score_ci_high": expected_ci_high,
                }

        global_result = outputs[0]["score"]["global"].copy()
        logger.info(global_result)
        for score_name, score_value in global_result.items():
            if score_name in expected_global_result:
                self.assertAlmostEqual(
                    expected_global_result[score_name],
                    score_value,
                    places=5,
                    msg=f"{score_name} score mismatch for {metric.__class__.__name__}, expected {expected_global_result[score_name]} but got {score_value}",
                )
            else:
                # An output score that is not expected
                # This is ok if the score_name is not related to confidence intervals
                # Otherwise, there was some confidence interval calculation that was not supposed to occur.
                self.assertTrue(
                    "ci_low" not in score_name and "ci_high" not in score_name,
                    msg=f"Unexpected confidence interval score '{score_name}'.",
                )

    def test_task_based_llm_as_judge_metric(self):
        model_id = "meta-llama/llama-3-8b-instruct"
        format = "formats.llama3_instruct"
        task = "tasks.rag_eval.answer_correctness.binary"
        template = "templates.rag_eval.answer_correctness.judge_loose_match_no_context"

        inference_model = MockInferenceEngine(
            model_name=model_id, default_inference_value="no"
        )
        model_label = inference_model.get_engine_id()
        template_label = template.split(".")[-1]
        metric_label = f"answer_correctness_{template_label}"
        metric = TaskBasedLLMasJudge(
            inference_model=inference_model,
            template=template,
            task=task,
            format=format,
            main_score=metric_label,
            infer_log_probs=False,
            include_meta_data=False,
        )

        predictions = [None, None]
        references = [[""], [""]]
        task_data = [
            {
                "question": "What foundation models are available in watsonx.ai ?",
                "answer": "Watsonx.ai supports no foundation models",
                "ground_truths": [
                    "Many Large Language Models are supported by Watsonx.ai"
                ],
                "contexts": ["Many Large Language Models are supported by Watsonx.ai"],
            }
        ] * 2

        outputs = apply_metric(
            metric=metric,
            predictions=predictions,
            references=references,
            task_data=task_data,
        )
        actual_scores = [output["score"] for output in outputs]
        main_score = f"{model_label}_{metric_label}"
        instance_targets = (
            [
                {
                    main_score: 0.0,
                    "score": 0.0,
                    "score_name": main_score,
                    main_score + "_judge_raw_output": "no",
                    main_score
                    + "_judge_raw_input": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are given a question, the corresponding ground-truth answer and a prediction from a model. Compare the "Ground-truth answer" and the "Prediction" to determine whether the prediction correctly answers the question.
There should be no contradicting statements in the prediction. The prediction may contain extra information. If the prediction states something as a possibility, treat it as a definitive answer.
The prediction must contain all the important information presented in the ground truths, but doesn't have to fully match it.
Answer with only yes/no.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: What foundation models are available in watsonx.ai ?

Ground-truth answer: Many Large Language Models are supported by Watsonx.ai

Prediction: Watsonx.ai supports no foundation models
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Answer: """,
                }
            ]
            * 2
        )
        global_target = {
            main_score: 0.0,
            "score": 0.0,
            "score_name": main_score,
            "num_of_instances": 2,
        }

        expected_scores = [
            {
                "global": global_target,
                "instance": instance_target,
            }
            for instance_target in instance_targets
        ]

        self.assertListEqual(actual_scores, expected_scores)

    def test_llm_as_judge_metric(self):
        model_id = "meta-llama/llama-3-8b-instruct"
        format = "formats.llama3_instruct"
        task = "rating.single_turn"
        template = "templates.response_assessment.rating.mt_bench_single_turn"

        inference_model = MockInferenceEngine(model_name=model_id)
        model_label = model_id.split("/")[1].replace("-", "_")
        model_label = f"{model_label}_ibm_genai"
        template_label = template.split(".")[-1]
        metric_label = f"{model_label}_template_{template_label}"
        metric = LLMAsJudge(
            inference_model=inference_model,
            template=template,
            task=task,
            format=format,
            main_score=metric_label,
        )

        predictions = ["[[10]]"] * 3
        references = [["[[10]]"], ["[[10]]"], ["[[10]]"]]
        task_data = [
            {
                "input": "input",
                "type_of_input": "type",
                "output": "output",
                "type_of_output": "type",
                "source": "<SYS_PROMPT>input</SYS_PROMPT>",
                "metadata": {
                    "template": "templates.generation.default",
                    "data_classification_policy": ["public"],
                },
            }
        ] * 3

        outputs = apply_metric(
            metric=metric,
            predictions=predictions,
            references=references,
            task_data=task_data,
        )
        actual_scores = [output["score"] for output in outputs]
        instance_targets = [
            {
                metric_label: 1.0,
                "score_name": metric_label,
                "score": 1.0,
                f"{metric_label}_judge_raw_input": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "Please act as an impartial judge and "
                "evaluate the quality of the response "
                "provided by an AI assistant to the user "
                "question displayed below. Your evaluation "
                "should consider factors such as the "
                "helpfulness, relevance, accuracy, depth, "
                "creativity, and level of detail of the "
                "response. Begin your evaluation by "
                "providing a short explanation. Be as "
                "objective as possible. After providing your "
                "explanation, you must rate the response on "
                "a scale of 1 to 10 by strictly following "
                'this format: "[[rating]]", for example: '
                '"Rating: [[5]]".\n\n'
                "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                "[Question]\n"
                "Given the following type, generate the corresponding type. type: input\n\n\n"
                "[The Start of Assistant's Answer]\n"
                "[[10]]\n"
                "[The End of Assistant's "
                "Answer]<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                f"{metric_label}_judge_raw_output": "[[10]]",
            }
        ] * 3
        global_target = {
            metric_label: 1.0,
            "score": 1.0,
            "score_name": metric_label,
            "num_of_instances": 3,
        }

        expected_scores = [
            {
                "global": global_target,
                "instance": instance_target,
            }
            for instance_target in instance_targets
        ]

        self.assertListEqual(actual_scores, expected_scores)

    def test_llm_as_judge_metric_with_chat_api(self):
        model_id = "meta-llama/llama-3-8b-instruct"
        format = "formats.chat_api"
        # format = "formats.llama3_instruct"
        task = "rating.single_turn"
        template = "templates.response_assessment.rating.mt_bench_single_turn"

        inference_model = MockInferenceEngine(model_name=model_id)
        model_label = model_id.split("/")[1].replace("-", "_")
        model_label = f"{model_label}_ibm_genai"
        template_label = template.split(".")[-1]
        metric_label = f"{model_label}_template_{template_label}"
        metric = LLMAsJudge(
            inference_model=inference_model,
            template=template,
            task=task,
            format=format,
            main_score=metric_label,
        )

        predictions = ["[[10]]"] * 3
        references = [["[[10]]"], ["[[10]]"], ["[[10]]"]]
        task_data = [
            {
                "input": "input",
                "type_of_input": "type",
                "output": "output",
                "type_of_output": "type",
                "source": "<SYS_PROMPT>input</SYS_PROMPT>",
                "metadata": {
                    "template": "templates.generation.default",
                    "data_classification_policy": ["public"],
                },
            }
        ] * 3

        outputs = apply_metric(
            metric=metric,
            predictions=predictions,
            references=references,
            task_data=task_data,
        )
        actual_scores = [output["score"] for output in outputs]
        instance_targets = [
            {
                metric_label: 1.0,
                "score_name": metric_label,
                "score": 1.0,
                f"{metric_label}_judge_raw_input": [
                    {
                        "role": "system",
                        "content": 'Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n',
                    },
                    {
                        "role": "user",
                        "content": "[Question]\nGiven the following type, generate the corresponding type. type: input\n\n\n[The Start of Assistant's Answer]\n[[10]]\n[The End of Assistant's Answer]",
                    },
                ],
                f"{metric_label}_judge_raw_output": "[[10]]",
            }
        ] * 3
        global_target = {
            metric_label: 1.0,
            "score": 1.0,
            "score_name": metric_label,
            "num_of_instances": 3,
        }

        expected_scores = [
            {
                "global": global_target,
                "instance": instance_target,
            }
            for instance_target in instance_targets
        ]

        self.assertListEqual(actual_scores, expected_scores)

    def test_fin_qa_eval(self):
        table = """[
            [
                "",
                "amount ( in millions )"
            ],
            [
                "2014 net revenue",
                "$ 5735"
            ],
            [
                "retail electric price",
                "187"
            ],
            [
                "volume/weather",
                "95"
            ],
            [
                "waterford 3 replacement steam generator provision",
                "-32 ( 32 )"
            ],
            [
                "miso deferral",
                "-35 ( 35 )"
            ],
            [
                "louisiana business combination customer credits",
                "-107 ( 107 )"
            ],
            [
                "other",
                "-14 ( 14 )"
            ],
            [
                "2015 net revenue",
                "$ 5829"
            ]
        ]"""

        table2 = """[
            [
                "statement of income classification",
                "statement of income loss on swaps",
                "statement of income gain on note",
                "statement of income net income effect",
                "statement of income gain on swaps",
                "loss on note",
                "net income effect"
            ],
            [
                "other income",
                "$ -4614 ( 4614 )",
                "$ 4614",
                "$ 2014",
                "$ 20692",
                "$ -20692 ( 20692 )",
                "$ 2014"
            ]
        ]"""

        metric = FinQAEval()
        references = [
            ["subtract(5829, 5735)"],
            ["subtract(5829, 5735)"],
            ["subtract(5829, 5735)"],
            ["subtract(5829, 5735)"],
            ["subtract(153.7, 139.9), divide(#0, 139.9)"],
        ]
        task_data = [
            {"table": table, "program_re": "subtract(5829, 5735)", "answer": "94"},
            {"table": table, "program_re": "subtract(5829, 5735)", "answer": "94"},
            {"table": table, "program_re": "subtract(5829, 5735)", "answer": "94%%"},
            {"table": table, "program_re": "subtract(5829, 5735)", "answer": "94"},
            {
                "table": table2,
                "program_re": "subtract(153.7, 139.9), divide(#0, 139.9)",
                "answer": "9.9%",
            },
        ]
        predictions = [
            "subtract(5829, 5735)",  # right program, right accuracy
            "subtract(5829, 5730)--",  # wrong program, wrong accuracy
            "subtract(5829, 5735)   ",  # answer with special chars (in task data)
            "subtract(5824, 5730), ",  # wrong program, right accuracy
            "subtract(153.7, 139.9), divide(#0, 139.9), ,",  # 2 operations
        ]

        outputs = apply_metric(
            metric=metric,
            predictions=predictions,
            references=references,
            task_data=task_data,
        )
        actual_scores = [
            (
                output["score"]["instance"]["program_accuracy"]
                + output["score"]["instance"]["execution_accuracy"]
            )
            / 2
            for output in outputs
        ]
        target_scores = [1, 0, 1, 0.5, 1]

        for i in range(len(actual_scores)):
            self.assertAlmostEqual(actual_scores[i], target_scores[i])

    def test_metrics_ensemble(self):
        metric = MetricsEnsemble(
            main_score="ensemble_score",
            metrics=[
                "metrics.precision_micro_multi_label",
                "metrics.recall_macro_multi_label",
            ],
            weights=None,
        )

        predictions = [["A"], ["B"], [""], ["A"]]
        references = [[["B", "A"]], [["B"]], [["A"]], [[""]]]

        instance_targets = [
            {
                "ensemble_score": 0.75,
                "ensemble_0_precision_micro": 1.0,
                "ensemble_1_recall_macro": 0.5,
                "score": 0.75,
                "score_name": "ensemble_score",
            },
            {
                "ensemble_score": 1.0,
                "ensemble_0_precision_micro": 1.0,
                "ensemble_1_recall_macro": 1.0,
                "score": 1.0,
                "score_name": "ensemble_score",
            },
            {
                "ensemble_score": 0.0,
                "ensemble_0_precision_micro": 0.0,
                "ensemble_1_recall_macro": 0.0,
                "score": 0.0,
                "score_name": "ensemble_score",
            },
            {
                "ensemble_score": 0.0,
                "ensemble_0_precision_micro": 0.0,
                "ensemble_1_recall_macro": 0.0,
                "score": 0.0,
                "score_name": "ensemble_score",
            },
        ]

        global_target = {
            "ensemble_0_precision_micro": 0.5,
            "ensemble_0_precision_micro_ci_high": 1.0,
            "ensemble_0_precision_micro_ci_low": 0.0,
            "ensemble_1_recall_macro": 0.33,
            "ensemble_1_recall_macro_ci_high": 0.56,
            "ensemble_1_recall_macro_ci_low": 0.0,
            "ensemble_score": 0.44,
            "score": 0.44,
            "score_name": "ensemble_score",
            "num_of_instances": 4,
        }

        test_metric(
            metric=metric,
            predictions=predictions,
            references=references,
            instance_targets=instance_targets,
            global_target=global_target,
        )

    def test_anls(self):
        metric = ANLS()

        predictions = ["A", "B", "C"]
        references = [["B"], ["A"], ["C"]]

        instance_targets = [
            {"anls": 0.0, "score": 0.0, "score_name": "anls"},
            {"anls": 0.0, "score": 0.0, "score_name": "anls"},
            {"anls": 1.0, "score": 1.0, "score_name": "anls"},
        ]

        global_target = {
            "anls": 0.33,
            "score": 0.33,
            "score_name": "anls",
            # "anls_ci_low": 0.0,
            # "anls_ci_high": 1.0,
            # "score_ci_low": 0.0,
            # "score_ci_high": 1.0,
            "num_of_instances": 3,
        }

        test_metric(
            metric=metric,
            predictions=predictions,
            references=references,
            instance_targets=instance_targets,
            global_target=global_target,
        )

    def test_websrc(self):
        metric = WebsrcSquadF1(n_resamples=None)

        predictions = ["The 2nd", "The 1st"]
        references = [["The 2nd"], ["The 2nd"]]

        # how to create a metric which isn't updated in every sample when using UNITXT?
        instance_targets = [
            {
                "websrc_squad_f1": 1.0,
                "score": 1.0,
                "score_name": "websrc_squad_f1",
            },
            {
                "websrc_squad_f1": 0.5,
                "score": 0.5,
                "score_name": "websrc_squad_f1",
            },
        ]
        global_target = {
            "num_of_instances": 2,
            "websrc_squad_f1": 0.75,
            "score": 0.75,
            "score_name": "websrc_squad_f1",
        }
        test_metric(
            metric=metric,
            predictions=predictions,
            references=references,
            instance_targets=instance_targets,
            global_target=global_target,
            task_data=[{"domain": "movie"}, {"domain": "movie"}],
        )

    def test_relaxed_correctness(self):
        metric = RelaxedCorrectness(n_resamples=None)

        predictions = ["10", "30"]
        references = [["14"], ["30"]]

        # how to create a metric which isn't updated in every sample when using UNITXT?
        instance_targets = [
            {
                "relaxed_overall": 0.0,
                "relaxed_human_split": 0.0,
                "score": 0.0,
                "score_name": "relaxed_overall",
            },
            {
                "relaxed_overall": 1.0,
                "relaxed_augmented_split": 1.0,
                "score": 1.0,
                "score_name": "relaxed_overall",
            },
        ]

        global_target = {
            "relaxed_overall": 0.5,
            "relaxed_human_split": 0.0,
            "relaxed_augmented_split": 1.0,
            "score": 0.5,
            "score_name": "relaxed_overall",
            "num_of_instances": 2,
        }
        test_metric(
            metric=metric,
            predictions=predictions,
            references=references,
            instance_targets=instance_targets,
            global_target=global_target,
            task_data=[{"type": "human_test"}, {"type": "augmented_test"}],
        )

    def test_string_containment(self):
        metric = StringContainment()

        predictions = [
            "barak obama is a politician",
            "David Gilmour is an English guitarist",
        ]
        references = [["politician", "politic", "pol", "musician"], ["artist"]]

        instance_targets = [
            {
                "string_containment": 1.0,
                "score": 1.0,
                "score_name": "string_containment",
            },
            {
                "string_containment": 0.0,
                "score": 0.0,
                "score_name": "string_containment",
            },
        ]

        global_target = {
            "string_containment": 0.50,
            "score": 0.50,
            "score_name": "string_containment",
            "score_ci_high": 1.0,
            "score_ci_low": 0.0,
            "string_containment_ci_high": 1.0,
            "string_containment_ci_low": 0.0,
            "num_of_instances": 2,
        }

        test_metric(
            metric=metric,
            predictions=predictions,
            references=references,
            instance_targets=instance_targets,
            global_target=global_target,
        )

        reference_field = "entities"

        metric = StringContainmentRatio(field=reference_field)

        instance_targets = [
            {
                "string_containment": 0.75,
                "score": 0.75,
                "score_name": "string_containment",
            },
            {
                "string_containment": 0.0,
                "score": 0.0,
                "score_name": "string_containment",
            },
        ]

        global_target = {
            "string_containment": 0.38,
            "score": 0.38,
            "score_name": "string_containment",
            "score_ci_high": 0.75,
            "score_ci_low": 0.0,
            "string_containment_ci_high": 0.75,
            "string_containment_ci_low": 0.0,
            "num_of_instances": 2,
        }

        test_metric(
            metric=metric,
            predictions=predictions,
            references=[["dummy"] for _ in references],
            instance_targets=instance_targets,
            global_target=global_target,
            task_data=[{reference_field: w} for w in references],
        )

    def test_meteor(self):
        import nltk

        nltk.download("punkt_tab", quiet=True)
        metric = MeteorFast(
            __description__="""METEOR (Metric for Evaluation of Translation with Explicit ORdering) is a machine translation evaluation metric, which is calculated based on the harmonic mean of precision and recall, with recall weighted more than precision.

        METEOR is based on a generalized concept of unigram matching between the machine-produced translation and human-produced reference translations. Unigrams can be matched based on their surface forms, stemmed forms, and meanings. Once all generalized unigram matches between the two strings have been found, METEOR computes a score for this matching using a combination of unigram-precision, unigram-recall, and a measure of fragmentation that is designed to directly capture how well-ordered the matched words in the machine translation are in relation to the reference.
        """,
        )

        predictions = [
            "It is a guide to action which ensures that the military always obeys the commands of the party",
            "We strive for peace",
            "On the rag sat the cat",
            "I caught the ball",
        ]
        references = [
            [
                "It is a guide to action that ensures that the military will forever heed Party commands"
            ],
            ["We hope for peace"],
            ["The cat sat on the rag"],
            ["He threw the ball"],
        ]

        # the floats shown here are rounded just for the test. the actually
        # returned score are 15-16 digits to the right of the decimal point
        instance_targets = [
            {"meteor": 0.69, "score": 0.69, "score_name": "meteor"},
            {"meteor": 0.64, "score": 0.64, "score_name": "meteor"},
            {"meteor": 0.5, "score": 0.5, "score_name": "meteor"},
            {"meteor": 0.47, "score": 0.47, "score_name": "meteor"},
        ]

        global_target = {
            "meteor": 0.58,
            "meteor_ci_high": 0.67,
            "meteor_ci_low": 0.48,
            "num_of_instances": 4,
            "score": 0.58,
            "score_ci_high": 0.67,
            "score_ci_low": 0.48,
            "score_name": "meteor",
        }

        test_metric(
            metric=metric,
            predictions=predictions,
            references=references,
            instance_targets=instance_targets,
            global_target=global_target,
        )

        # to match the setting to occur by testing on the global version, metric2, below, setting n_resamples=3

        metric_hf = MeteorFast(
            n_resamples=3,
            __description__="""Huggingface version with bad confidence interval calculation of METEOR (Metric for Evaluation of Translation with Explicit ORdering) is a machine translation evaluation metric, which is calculated based on the harmonic mean of precision and recall, with recall weighted more than precision.

        METEOR is based on a generalized concept of unigram matching between the machine-produced translation and human-produced reference translations. Unigrams can be matched based on their surface forms, stemmed forms, and meanings. Once all generalized unigram matches between the two strings have been found, METEOR computes a score for this matching using a combination of unigram-precision, unigram-recall, and a measure of fragmentation that is designed to directly capture how well-ordered the matched words in the machine translation are in relation to the reference.
        """,
        )

        global_target = {
            "meteor": 0.58,
            "meteor_ci_high": 0.59,
            "meteor_ci_low": 0.58,
            "score": 0.58,
            "score_ci_high": 0.59,
            "score_ci_low": 0.58,
            "score_name": "meteor",
            "num_of_instances": 4,
        }

        test_metric(
            metric=metric_hf,
            predictions=predictions,
            references=references,
            instance_targets=instance_targets,
            global_target=global_target,
        )

        # compare results with the HF version of meteor
        metric2 = HuggingfaceMetric(
            hf_metric_name="meteor", main_score="meteor", prediction_type=str
        )

        test_metric(
            metric=metric2,
            predictions=predictions,
            references=references,
            instance_targets=instance_targets,
            global_target=global_target,
        )

    def test_char_edit_distance(self):
        metric = CharEditDistanceAccuracy()
        abs_dist_metric = CharEditDistance()

        predictions = ["this is the prediction", "there is an other sample"]
        references = [["this is the reference"], ["there is another sample"]]

        # First sample:   p[re]diction - edit distance (8), max len ignoring whitespace (19)  accuracy = 1 - 8/19 = 0.578
        # Second sample: [an other] [another] - edit distance ignoring white space(0), max len ignoring whitespace (19)     accuracy = 1 - 0/19 = 1

        instance_targets = [
            {
                "char_edit_dist_accuracy": 0.58,
                "score": 0.58,
                "score_name": "char_edit_dist_accuracy",
            },
            {
                "char_edit_dist_accuracy": 1.00,
                "score": 1.00,
                "score_name": "char_edit_dist_accuracy",
            },
        ]

        global_target = {
            "char_edit_dist_accuracy": 0.79,
            "score": 0.79,
            "score_name": "char_edit_dist_accuracy",
            "char_edit_dist_accuracy_ci_low": 0.58,
            "char_edit_dist_accuracy_ci_high": 1.0,
            "score_ci_low": 0.58,
            "score_ci_high": 1.0,
            "num_of_instances": 2,
        }

        test_metric(
            metric=metric,
            predictions=predictions,
            references=references,
            instance_targets=instance_targets,
            global_target=global_target,
        )

        dist_instance_targets = [
            {
                "char_edit_distance": 8,
                "score": 8,
                "score_name": "char_edit_distance",
            },
            {
                "char_edit_distance": 0,
                "score": 0,
                "score_name": "char_edit_distance",
            },
        ]

        dist_global_target = {
            "char_edit_distance": 4.0,
            "score": 4.0,
            "score_name": "char_edit_distance",
            "char_edit_distance_ci_low": 0.0,
            "char_edit_distance_ci_high": 8.0,
            "score_ci_low": 0.0,
            "score_ci_high": 8.0,
            "num_of_instances": 2,
        }

        test_metric(
            metric=abs_dist_metric,
            predictions=predictions,
            references=references,
            instance_targets=dist_instance_targets,
            global_target=dist_global_target,
        )

        predictions = [""]
        references = [[""]]

        instance_targets = [
            {
                "char_edit_dist_accuracy": 0.0,
                "score": 0.0,
                "score_name": "char_edit_dist_accuracy",
            }
        ]

        global_target = {
            "char_edit_dist_accuracy": 0.0,
            "score": 0.0,
            "score_name": "char_edit_dist_accuracy",
            "num_of_instances": 1,
        }

        test_metric(
            metric=metric,
            predictions=predictions,
            references=references,
            instance_targets=instance_targets,
            global_target=global_target,
        )

        dist_instance_targets = [
            {
                "char_edit_distance": 0.0,
                "score": 0.0,
                "score_name": "char_edit_distance",
            }
        ]

        dist_global_target = {
            "char_edit_distance": 0.0,
            "score": 0.0,
            "score_name": "char_edit_distance",
            "num_of_instances": 1,
        }

        test_metric(
            metric=abs_dist_metric,
            predictions=predictions,
            references=references,
            instance_targets=dist_instance_targets,
            global_target=dist_global_target,
        )

    def test_rouge_new(self):
        import nltk

        nltk.download("punkt_tab", quiet=True)
        metric = Rouge(
            __description__="""This is the classical NLP Rouge metric based on the RougeScorer library (https://github.com/google-research/google-research/tree/master/rouge).
        It computes metrics several metrics (rouge1, rouge2, roughL, and rougeLsum) based lexical (word) overlap between the prediction and the ground truth references."
        """,
            __tags__={"flags": ["reference-based-metric", "cpu-metric"]},
        )
        predictions = ["hello there", "general kenobi"]
        references = [["hello", "there"], ["general kenobi", "general yoda"]]

        instance_targets = [
            {
                "rouge1": 0.67,
                "rouge2": 0.0,
                "rougeL": 0.67,
                "rougeLsum": 0.67,
                "score": 0.67,
                "score_name": "rougeL",
            },
            {
                "rouge1": 1.0,
                "rouge2": 1.0,
                "rougeL": 1.0,
                "rougeLsum": 1.0,
                "score": 1.0,
                "score_name": "rougeL",
            },
        ]

        global_target = {
            "rouge1": 0.83,
            "rouge1_ci_high": 1.0,
            "rouge1_ci_low": 0.67,
            "rouge2": 0.5,
            "rouge2_ci_high": 1.0,
            "rouge2_ci_low": 0.0,
            "rougeL": 0.83,
            "rougeL_ci_high": 1.0,
            "rougeL_ci_low": 0.67,
            "rougeLsum": 0.83,
            "rougeLsum_ci_high": 1.0,
            "rougeLsum_ci_low": 0.67,
            "score": 0.83,
            "score_ci_high": 1.0,
            "score_ci_low": 0.67,
            "score_name": "rougeL",
            "num_of_instances": 2,
        }
        test_metric(
            metric=metric,
            predictions=predictions,
            references=references,
            instance_targets=instance_targets,
            global_target=global_target,
        )

    def test_ndcg(self):
        import numpy as np
        from unitxt.blocks import CastFields
        from unitxt.metrics import NDCG, MetricPipeline

        # Normalized Discounted Cumulative Gain
        metric = MetricPipeline(
            main_score="nDCG",
            single_reference_per_prediction=True,
            preprocess_steps=[
                CastFields(
                    fields={"prediction": "float", "references/0": "float"},
                    failure_defaults={"prediction": None},
                ),
            ],
            metric=NDCG(),
        )

        predictions = [
            "1.0",
            " 2 ",
            "1.0",
            "0",
            "1.7",
            3,
            "0",
            "oops",
            "1",
            "failed",
            "failed again",
        ]
        references = [
            ["4"],
            ["0"],
            ["1.0"],
            [4],
            ["0"],
            ["1"],
            ["1.0"],
            ["3"],
            ["2"],
            [4],
            [1],
        ]
        inputs = (
            [{"query": "who is Barack Obama"}] * 3
            + [{"query": "What is an albatross?"}] * 5
            + [{"query": "something else"}]
            + [{"query": "these will fail"}] * 2
        )
        instance_targets = [  # nDCG is undefined at instance level
            {"nDCG": np.nan, "score": np.nan, "score_name": "nDCG"}
        ] * len(predictions)

        global_target = {
            "nDCG": 0.42,
            "nDCG_ci_high": 0.66,
            "nDCG_ci_low": 0.15,
            "score": 0.42,
            "score_ci_high": 0.66,
            "score_ci_low": 0.15,
            "score_name": "nDCG",
            "num_of_instances": 11,
        }
        test_metric(
            metric=metric,
            predictions=predictions,
            references=references,
            task_data=inputs,
            instance_targets=instance_targets,
            global_target=global_target,
        )

    def test_llm_as_judge(self):
        data = [
            {"question": "Who is Harry Potter?"},
            {
                "question": "How can I protect myself from the wind while walking outside?"
            },
            {"question": "What is a good low cost of living city in the US?"},
        ]

        criterion = "metrics.llm_as_judge.direct.criteria.answer_relevance"
        metrics = [
            f"metrics.llm_as_judge.direct.rits.llama3_3_70b[criteria={criterion}, context_fields=[question]]"
        ]

        dataset = create_dataset(
            task="tasks.qa.open", test_set=data, metrics=metrics, split="test"
        )

        predictions = [
            """Harry Potter is a young wizard who becomes famous for surviving an attack by the dark wizard Voldemort, and later embarks on a journey to defeat him and uncover the truth about his past.""",
            """You can protect yourself from the wind by wearing windproof clothing, layering up, and using accessories like hats, scarves, and gloves to cover exposed skin.""",
            """A good low-cost-of-living city in the U.S. is San Francisco, California, known for its affordable housing and budget-friendly lifestyle.""",
        ]

        results = evaluate(predictions=predictions, data=dataset)
        self.assertDictEqual(
            dict(results[0]["score"]["global"]),
            {
                "num_of_instances": 3,
                "answer_relevance": 0.5,
                "score": 0.5,
                "score_name": "answer_relevance",
            },
        )
