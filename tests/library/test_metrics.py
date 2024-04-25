from math import isnan

from unitxt.inference import HFPipelineBasedInferenceEngine
from unitxt.llm_as_judge import LLMAsJudge
from unitxt.logging_utils import get_logger
from unitxt.metrics import (
    NER,
    Accuracy,
    BinaryAccuracy,
    BinaryMaxAccuracy,
    BinaryMaxF1,
    Detector,
    F1Binary,
    F1Macro,
    F1MacroMultiLabel,
    F1Micro,
    F1MicroMultiLabel,
    F1Weighted,
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
    GroupMeanAccuracy,
    GroupMeanStringContainment,
    GroupMeanTokenOverlap,
    KendallTauMetric,
    LlamaIndexCorrectness,
    MaxAccuracy,
    NormalizedSacrebleu,
    Perplexity,
    PrecisionBinary,
    RecallBinary,
    RocAuc,
    Rouge,
    TokenOverlap,
    UnsortedListExactMatch,
)
from unitxt.test_utils.metrics import apply_metric

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
        references = [["1"], ["0"], ["0"], ["0"], ["Yes"], ["1"]]
        predictions = ["0.8", "1", "0.2", "0", "0.6", "1"]

        global_target = 0.8571428571428
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )

        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        self.assertEqual("f1_binary", outputs[0]["score"]["global"]["score_name"])
        self.assertEqual("f1_binary", outputs[0]["score"]["instance"]["score_name"])

    def test_precision_binary(self):
        metric = PrecisionBinary()
        references = [["1"], ["0"], ["0"], ["0"], ["1"], ["1"]]
        predictions = ["1", "1", "0", "0", "1", "1"]

        global_target = 0.75
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )

        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        self.assertEqual(
            "precision_binary", outputs[0]["score"]["global"]["score_name"]
        )
        self.assertEqual(
            "precision_binary", outputs[0]["score"]["instance"]["score_name"]
        )

    def test_recall_binary(self):
        metric = RecallBinary()
        references = [["1"], ["0"], ["0"], ["0"], ["1"], ["1"]]
        predictions = ["1", "1", "0", "0", "1", "1"]

        global_target = 1
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )

        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        self.assertEqual("recall_binary", outputs[0]["score"]["global"]["score_name"])
        self.assertEqual("recall_binary", outputs[0]["score"]["instance"]["score_name"])

    def test_max_f1(self):
        metric = BinaryMaxF1()
        references = [["1"], ["0"], ["0"]]
        predictions = ["0.3", "0", "0.7"]

        global_target = 0.666666666666
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )

        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        self.assertEqual("max_f1_binary", outputs[0]["score"]["global"]["score_name"])
        self.assertEqual("max_f1_binary", outputs[0]["score"]["instance"]["score_name"])

    def test_accuracy_binary(self):
        metric = BinaryAccuracy()
        references = [["1"], ["0"], ["0"], ["1"], ["0"]]
        predictions = ["0.3", "0", "0.7", "1.0", "0.2"]

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
        references = [["1"], ["0"], ["0"], ["1"], ["0"]]
        predictions = ["0.3", "0", "0.7", "1.0", "0.2"]

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

        references = [["0"], ["0"], ["0"]]
        predictions = ["0.3", "0.9", "0.7"]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(1.0, outputs[0]["score"]["global"]["score"])

        references = [["1"], ["0"], ["0"], ["1"], ["0"], ["0"]]
        predictions = ["0.7", "0.3", "0.7", "0.8", "0.9", "0.3"]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(2 / 3, outputs[0]["score"]["global"]["score"])

        references = [["1"]]
        predictions = ["0.7"]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(1.0, outputs[0]["score"]["global"]["score"])

        references = [["0"]]
        predictions = ["0.7"]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        self.assertAlmostEqual(1.0, outputs[0]["score"]["global"]["score"])

        references = [["0"]]
        predictions = ["1.7"]
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

    def test_rouge(self):
        metric = Rouge()
        references = [["hello", "there"], ["general kenobi", "general yoda"]]
        predictions = ["hello there", "general kenobi"]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        global_target = 5 / 6
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])

    def test_rouge_l(self):
        metric = Rouge(
            n_resamples=None,  # disable confidence interval calculation which fails for this metric configuration
            use_aggregator=False,
            rouge_types=["rougeL"],
        )
        references = [["hello", "there"], ["general kenobi", "general yoda"]]
        predictions = ["hello there", "general kenobi"]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        global_target = [2 / 3, 1.0]
        self.assertListEqual(global_target, outputs[0]["score"]["global"]["score"])

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
        predictions = ["0.2", "0.8", "1.0"]
        references = [["1.0"], ["0.0"], ["1.0"]]
        outputs = apply_metric(
            metric=metric, predictions=predictions, references=references
        )
        global_target = 0.5
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])

    def test_kendalltau(self):
        metric = KendallTauMetric()
        predictions = ["1.0", "2.0", "1.0"]
        references = [["-1.0"], ["1.0"], ["0.0"]]
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

    def test_perplexity(self):
        prediction = ["who are we?"]
        references = [["we are the world"]]

        perplexity_question = Perplexity(
            model_name="google/flan-t5-small",
            perplexity_prompt="Generate a question based on the given content:",
        )
        first_instance_target = 0.059865921735763
        outputs = apply_metric(
            metric=perplexity_question, predictions=prediction, references=references
        )
        self.assertAlmostEqual(
            first_instance_target, outputs[0]["score"]["instance"]["score"]
        )

        perplexity_question_mistral = Perplexity(
            model_name="google/flan-t5-small",
            perplexity_prompt="Generate a question based on the given content: %s",
        )
        outputs = apply_metric(
            metric=perplexity_question_mistral,
            predictions=prediction,
            references=references,
        )
        self.assertAlmostEqual(
            first_instance_target, outputs[0]["score"]["instance"]["score"]
        )


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
                    score_value, expected_global_result[score_name], places=5
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
            expected_ci_high=0.44105968464125495,
        )

        self._test_grouped_instance_confidence_interval(
            metric=FixedGroupMeanStringContainment(),
            expected_ci_low=0.0,
            expected_ci_high=0.675,
        )

        self._test_grouped_instance_confidence_interval(
            metric=GroupMeanStringContainment(),
            expected_ci_low=0.15556138609239942,
            expected_ci_high=0.707936507936508,
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
        self._test_grouped_instance_confidence_interval(
            metric=GroupMeanTokenOverlap(),
            expected_global_result={
                "group_mean_recall": 0.525,
                "group_mean_f1": 0.5083333333333333,
                "score": 0.5083333333333333,
                "score_name": "group_mean_f1",
                "group_mean_precision": 0.5,
                "group_mean_recall_ci_low": 0.25,
                "group_mean_recall_ci_high": 0.7083333333333334,
                "group_mean_f1_ci_low": 0.22302503471948287,
                "group_mean_f1_ci_high": 0.6805555555555555,
                "score_ci_low": 0.22302503471948287,
                "score_ci_high": 0.6805555555555555,
                "group_mean_precision_ci_low": 0.20949399775845196,
                "group_mean_precision_ci_high": 0.6666666666666666,
            },
        )

    def _test_grouped_instance_confidence_interval(
        self,
        metric,
        expected_ci_low=0.0,
        expected_ci_high=1.0,
        expected_global_result=None,
    ):
        """Test the calculation of confidence intervals for a given metric with group_mean reduction."""
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
                metric.main_score,
            ]
        )

        if expected_global_result is None:
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
                    score_value,
                    expected_global_result[score_name],
                    places=5,
                    msg=f"{group_score_name} score mismatch for {metric.__class__.__name__}, got {expected_global_result[score_name]} but expected {score_value}",
                )
            else:
                # An output score that is not expected
                # This is ok if the score_name is not related to confidence intervals
                # Otherwise, there was some confidence interval calculation that was not supposed to occur.
                self.assertTrue(
                    "ci_low" not in score_name and "ci_high" not in score_name,
                    msg=f"Unexpected confidence interval score '{score_name}'.",
                )

    def test_llm_as_judge_metric(self):
        inference_model = HFPipelineBasedInferenceEngine(
            model_name="google/flan-t5-small", max_new_tokens=32
        )
        recipe = (
            "card=cards.rag.model_response_assessment.llm_as_judge_using_mt_bench_template,"
            "template=templates.rag.model_response_assessment.llm_as_judge_using_mt_bench_template,"
            "demos_pool_size=0,"
            "num_demos=0"
        )

        metric = LLMAsJudge(inference_model=inference_model, recipe=recipe)

        predictions = [
            "Meditation can help you remember things in a more meaningful way. It can also help you remember things that you didn't know.",
            "Remove the fan from the fan and wipe it down with a damp cloth.",
            "Place a small amount of rubbing alcohol on the bag and rub it over the smell.",
            "Place the tank in the ground and place the guppy tank in the ground.",
            "Use a hair dryer to remove the chemical burns.",
        ]
        references = [
            [
                "Meditation has been scientifically proven to increase focus and memory. You don't have to use any one meditation to help your memory. Using any meditation, such as mindfulness meditation, teaches you to focus your mind. When you're able to focus better, you're also better able to solidify concepts in your short-term memory. Therefore, practicing meditation can help you to develop your short-term memory.\n1. **Start today.** You may be surprised that you don't need to practice meditation for that long to start seeing the effects. One scientific study examined how a group of students responded to meditation. With just two weeks of meditation practice (10 minutes a day, plus 4 45-minute classes a week), the students significantly improved their GRE scores (a standardized test given to students trying to get into graduate school).\nIn fact, some studies show as little as four days of meditation can improve your attention span and memory.\n2. **Practice often.** Practicing every day is ideal. Doing so will help you work to increase your memory. In fact, spreading it out throughout the day can be helpful, such as meditating for 10 minutes in the morning, 10 minutes at lunch, and 10 minutes in the evening. However, if you find you can't practice every day, do it as often as you can.\n3. **Cultivate mindfulness.** Mindfulness is a part of meditation, but it's also something you can incorporate in your day-to-day life. Mindfulness, at its most basic, just means paying attention. In other words, place yourself in the moment, rather then letting your mind race elsewhere.\n\nFor instance, when you're in the shower, stop yourself from thinking about the day ahead. Instead, focus on what the shower feels like. Feel the heat of the water on your skin, how the soap feels against your body. Pay attention to the pleasant scent of your soap or shampoo. Let yourself really feel each sensation.\nYou can practice this technique anywhere. For instance, while you're washing dishes, take a moment to really focus on what you're doing. Let yourself feel the warm water on your skin, the weight of a plate in your hands. Put your full attention on getting the plate clean, making sure it's spotless.\n4. **Work your way up.** You may want to jump in with an hour-long meditation every day. However, most people can't sustain that kind of practice when they haven't meditated before. It's best to start small and work up to more time. You can start with as little as three minutes a day.\n5. **Pick a place to meditate.** Really, you can meditate anywhere, but it's good to choose a place that's not distracting, particularly when you're first starting out. Turn off the television, and move away from distractions. You can even set up a little meditation center in a corner of your house, with a candle and something you like to focus on.\n6. **Sit properly.** You can sit in a chair or on the floor. It's up to you. However, make sure you are relatively comfortable. You don't want a lot of pressure on one part of your body, for instance. Try to sit up straight, though not so much that it feels like a strain.\n7. **Get settled.** Spend a few minutes just bringing yourself into the right state of mind. Focus on the candle, if that helps. You don't have to be completely focused, but as you feel your mind wander, bring it back to the center, to the moment.\n8. **Focus on your breathing.** Once you've situated yourself, try paying attention to just your breathing. Focus on it going in and out. You don't have to change it up. Rather, just keep your attention on it, focusing all of yourself on breathing in and out. As your mind wanders, bring it back to your breath.\n9. **Keep bringing yourself back.** The longer you sit, the more likely your mind is to wander. That's okay. It's normal in fact. The important thing is to acknowledge that you've wandered and move back to your focus. Try labeling it when your mind wanders, such as saying \"thinking\" in your head, and then refocusing on your breath.\n10. **Try deep breathing.** One simple way to get started with meditation is to try deep breathing. Start by placing a hand on your chest and a hand on your stomach. When you breathe, you should notice your stomach expanding more than your chest, as you are trying to breathe as deeply as possible. It can help to close your eyes. Breathe in slowly through your nose. Hold the breath to the count of seven, then let it slowly out through your mouth to the count of eight (in your head).\n\nTry taking five deep breaths each time you try this practice.\nMake sure you are blowing out fully.\n11. **Consider taking a class.** While classes aren't for everyone, a class can jump start your meditation practice, making it easier for you to make it an everyday practice. Plus, if you have no idea where to begin, a class will help you figure out a good starting point.\n\nLook for meditation centers in your area. Some yoga studios offer meditation classes as well. Also, Buddhist temples or centers in your area will likely offer classes on meditation.\nYou may also find meditation classes through your library or your local parks and recreation department, and some churches offer meditation classes, particularly ones that embrace other traditions, such as the Unitarian Universalists.\n12. **Don't let distraction make you anxious.** Everyone gets distracted when they meditate. When you're first starting out, that may make you anxious or angry at yourself. However, rather than becoming angry, just try to be aware of when your thoughts are drifting, and pull them back to the meditation.\n13. **Realize even a little meditation can help.** That is, you may think you have to meditate every single day at a certain time for it to be helpful. However, if you fall into that thinking, you may find yourself giving up because you miss a few days. Keep in mind that even a little meditation can help improve your memory. Therefore, try to meditate when you can, even if you don't find time to do it every day.\n14. **Try a guided meditation.** If you don't want to take a class, you can still benefit from the wisdom of others. Try doing a guided meditation. You can find many online, or you can download free apps. The person on the other end will walk you through a meditation process, helping you to learn how to do it.\n15. **Change it up.** You don't have to meditate the same way every time. For instance, some people find a walking meditation helpful. Take a ten-minute walk, focusing on different sensations in turn. Start with feeling your body walking, really focusing on what the movements feel like. Move on to the feeling of breathing. After that, focus on what the air feels like on your skin, then try thinking about just what you see and then just what you hear."
            ],
            [
                "One of the most neglected spaces when it comes to cleaning a bathroom is the fan. Having a clean, functional fan can lessen bathroom odors, as well as combat mold and mildew growth. These issues can become a health hazard if left unattended for too long. By cleaning your fan around every 6 months, you will be able to remove built up dirt before it becomes a problem.\n1. **Turn off the power.** Before you do anything else, ensure that the fan is turned off and cannot turn back on until you are finished cleaning it. Most models will have a plug that is located directly behind the cover. You could remove the cover first and unplug the fan, but just to be safe, go and temporarily pull the breaker for your bathroom. The fan is now safe to work on.\n2. **Remove the cover.** Dust will fall when the cover is removed. To avoid the dust, position your stepladder such that you can reach the cover, but are not standing directly below it. Most covers will have 2 prongs on opposite sides holding it in place, others just need to be unscrewed. Remove the cover by pressing these prongs in or removing the screws, then set the cover aside.\n3. **Remove the fan.** Unscrew the assembly that is holding the fan in place, then very gently remove the fan. Be careful not to drop the fan or hit it on the side of the exhaust pipe as that could potentially chip the fan blades. Broken fan blades will cause the fan to be louder and less effective.\n4. **Clean the cover and fan.** Start by vacuuming off the majority of the built up grime on both the cover and the fan. Then dip a rag, preferably a microfiber cloth, in soapy water and use it to wipe up the remaining dust. Be as thorough as you can, you will probably not do this again for a while.\nYou can let the cover soak in a tub of hot soapy water, but the fan should be wiped by hand to avoid getting water on the motor assembly or plug.\n5. **Vacuum the exhaust pipe.** Use a crevice or brush attachment and vacuum off the inside of the exhaust pipe. If you can reach, also use your rag or cloth to wipe off what the vacuum could not get.\n6. **Vacuum the external exhaust port.** This can be done later once the entire process is finished, but at some point you should go outside and find the exterior vent for your bathroom fan. Depending on where the bathroom is located, this vent will either be on the roof or the side of your house. Bring a damp rag to wipe off any dirt that has built up on the other end of your exhaust pipe.\n7. **Wipe and vacuum the fan housing.** If your fan had an accessible plug, be careful not to get any water inside the outlet. Doing so could result in electrocution or short circuit the fan when you plug it back in.  Therefore, use a dry rag to wipe off the fan housing, then vacuum up any remaining dust or debris.\n8. **Put the fan back in place.** Before reinstalling the fan, make sure that you cleaned off all the dust from in between each of the blades and dried it thoroughly. Carefully reinsert it into the exhaust pipe and screw the bracing back into place. Use your fingers and spin the fan around a few rotations to make sure that it is not rubbing against anything.\n9. **Turn the power back on.** Plug the fan back into the outlet and reset the breaker for your bathroom. The fan is now dangerous again, so do not touch it or continue to clean it after this point.\n10. **Reinstall the cover.** Once the cover has dried, either screw it back in or bend the prongs until the cover snaps back into place.\n11. **Test the fan.** Turn the fan on again to make sure everything works as normal. The fan should be quieter than it was before and provide a higher amount of air flow."
            ],
            [
                "Musty, stinky, odorous old leather bags aren't much fun and it's probable you're not keen to reuse such a bag. Before you resort to throwing it out, there are various ways that might just restore it to a respectable odor again.\n1. **Try a simple clean first.** If this clean doesn't shift the odor, you can try one of the other suggested methods after.\n\nWipe the leather bag inside and out with a clean, dry, soft cloth. This will pick up dust, loose debris and even some mold or mildew.\nWipe the leather bag down with a damp cloth. This will collect even more of the above items.\n2. **Allow the bag to air out.** Choose somewhere outdoors that is sheltered from direct light and heat, such as a table on the porch. Leave for a day if possible.\n3. **Check the odor.** If the bag still smells bad, choose one of the remaining suggested methods, or a combination of the methods.\n4. **Prepare a solution consisting of equal parts of white vinegar and distilled water.** Sponge the bag with the solution. Work on the inside of the bag and any mildewed outside part of the bag for a few minutes.\nIt's a good idea to test a small spot before trying this method, in case it stains.\n5. **Wipe off the vinegar solution with a clean, damp cloth.** \n6. **Allow to air dry.** Place the bag outside under shelter away from direct light to air dry.\n7. **Check the odor.** If it is still there, repeat. If not, the bag can be used again.\n8. **Use liquid detergent soap to clean the bag.** \n9. **Make a solution of soapy water, using the liquid detergent.** Dip the cleaning cloth or sponge in the solution and wring out before using.\n10. **Wipe the cloth over and inside the bag.** Concentrate in particular on the areas that you think are the smelliest.\n11. **Allow to air dry.** Place outside in a sheltered area away from direct sunlight and heat.\n12. **Once dry, check for the odor.** If it lingers, try again.\n13. **Use baking soda to deodorize the bag.** \n14. **Fill a clean sock with baking soda.** Tie off with a knot.\n15. **Place the leather bag and the baking soda-filled sock inside a large resealable plastic bag.** Alternatively, place both items inside an airtight container.\n16. **Set aside.** Let the baking soda work on the bag for at least 24 hours. The odors from the bag should transfer across to the baking soda.\n17. **Remove from the resealable bag or container.** Check the odor of the leather bag; if it still smells bad, repeat the process for another 24 hours, or longer. If it smells good again, throw away the baking soda, wash the sock and use the leather bag again.\n18. **Find some newspaper.** Scrunch the pages up and stuff them inside a large plastic bag, such as a kitchen waste bag or a garbage bag.\n19. **Slide the smelly leather bag in with the newspapers.** Arrange it so that it sits snugly in the middle of the papers.\n20. **Tie the bag up with a knot.** Alternatively, seal with a twist tie.\n21. **Let sit for at least 48 hours.** A few days more won't hurt it.\n22. **Remove from the bag.** Do a sniff test to see whether the odor has gone. If not, return to the bag for a few more days. Eventually it should start to smell better.\n23. **Fill a sock with coffee grounds.** They must be dry grounds, so if you're using grounds from your own coffee making, allow them to fully dry first. Or use the cheap instant coffee granules. Knot it off to keep the coffee intact.\n24. **Place the coffee sock inside your old leather bag.** Leave it there for up to a week. During this time, it should soak up much, if not all, of the cigarette smoke odor.\n25. **Do a smell test.** If all is good, the bag is ready for reuse. If it still smells a little, return the sock for a few more days.\n26. **Make or purchase some potpourri.** Place the potpourri inside a sachet.\n27. **Place the sachet inside the smelly bag.** Leave it there for at least one week.\n28. **Place the bag in an airy place.** Do not leave it in a dark cupboard; instead find somewhere with fresh air and indirect, cool light.\n29. **Check a week later.** It's a good idea to leave the sachet in the bag when using as well, as the scent will continue to improve the bag's own scent."
            ],
            ["Caring for guppies is relatively easy"],
            [
                "Many people suffer from hair that is damaged or burnt by various harsh chemical"
            ],
        ]
        task_data = [
            {
                "question": "How to Improve Your Memory Using Meditation",
                "answers": [
                    "Meditation has been scientifically proven to increase focus and memory. You don't have to use any one meditation to help your memory. Using any meditation, such as mindfulness meditation, teaches you to focus your mind. When you're able to focus better, you're also better able to solidify concepts in your short-term memory. Therefore, practicing meditation can help you to develop your short-term memory.\n1. **Start today.** You may be surprised that you don't need to practice meditation for that long to start seeing the effects. One scientific study examined how a group of students responded to meditation. With just two weeks of meditation practice (10 minutes a day, plus 4 45-minute classes a week), the students significantly improved their GRE scores (a standardized test given to students trying to get into graduate school).\nIn fact, some studies show as little as four days of meditation can improve your attention span and memory.\n2. **Practice often.** Practicing every day is ideal. Doing so will help you work to increase your memory. In fact, spreading it out throughout the day can be helpful, such as meditating for 10 minutes in the morning, 10 minutes at lunch, and 10 minutes in the evening. However, if you find you can't practice every day, do it as often as you can.\n3. **Cultivate mindfulness.** Mindfulness is a part of meditation, but it's also something you can incorporate in your day-to-day life. Mindfulness, at its most basic, just means paying attention. In other words, place yourself in the moment, rather then letting your mind race elsewhere.\n\nFor instance, when you're in the shower, stop yourself from thinking about the day ahead. Instead, focus on what the shower feels like. Feel the heat of the water on your skin, how the soap feels against your body. Pay attention to the pleasant scent of your soap or shampoo. Let yourself really feel each sensation.\nYou can practice this technique anywhere. For instance, while you're washing dishes, take a moment to really focus on what you're doing. Let yourself feel the warm water on your skin, the weight of a plate in your hands. Put your full attention on getting the plate clean, making sure it's spotless.\n4. **Work your way up.** You may want to jump in with an hour-long meditation every day. However, most people can't sustain that kind of practice when they haven't meditated before. It's best to start small and work up to more time. You can start with as little as three minutes a day.\n5. **Pick a place to meditate.** Really, you can meditate anywhere, but it's good to choose a place that's not distracting, particularly when you're first starting out. Turn off the television, and move away from distractions. You can even set up a little meditation center in a corner of your house, with a candle and something you like to focus on.\n6. **Sit properly.** You can sit in a chair or on the floor. It's up to you. However, make sure you are relatively comfortable. You don't want a lot of pressure on one part of your body, for instance. Try to sit up straight, though not so much that it feels like a strain.\n7. **Get settled.** Spend a few minutes just bringing yourself into the right state of mind. Focus on the candle, if that helps. You don't have to be completely focused, but as you feel your mind wander, bring it back to the center, to the moment.\n8. **Focus on your breathing.** Once you've situated yourself, try paying attention to just your breathing. Focus on it going in and out. You don't have to change it up. Rather, just keep your attention on it, focusing all of yourself on breathing in and out. As your mind wanders, bring it back to your breath.\n9. **Keep bringing yourself back.** The longer you sit, the more likely your mind is to wander. That's okay. It's normal in fact. The important thing is to acknowledge that you've wandered and move back to your focus. Try labeling it when your mind wanders, such as saying \"thinking\" in your head, and then refocusing on your breath.\n10. **Try deep breathing.** One simple way to get started with meditation is to try deep breathing. Start by placing a hand on your chest and a hand on your stomach. When you breathe, you should notice your stomach expanding more than your chest, as you are trying to breathe as deeply as possible. It can help to close your eyes. Breathe in slowly through your nose. Hold the breath to the count of seven, then let it slowly out through your mouth to the count of eight (in your head).\n\nTry taking five deep breaths each time you try this practice.\nMake sure you are blowing out fully.\n11. **Consider taking a class.** While classes aren't for everyone, a class can jump start your meditation practice, making it easier for you to make it an everyday practice. Plus, if you have no idea where to begin, a class will help you figure out a good starting point.\n\nLook for meditation centers in your area. Some yoga studios offer meditation classes as well. Also, Buddhist temples or centers in your area will likely offer classes on meditation.\nYou may also find meditation classes through your library or your local parks and recreation department, and some churches offer meditation classes, particularly ones that embrace other traditions, such as the Unitarian Universalists.\n12. **Don't let distraction make you anxious.** Everyone gets distracted when they meditate. When you're first starting out, that may make you anxious or angry at yourself. However, rather than becoming angry, just try to be aware of when your thoughts are drifting, and pull them back to the meditation.\n13. **Realize even a little meditation can help.** That is, you may think you have to meditate every single day at a certain time for it to be helpful. However, if you fall into that thinking, you may find yourself giving up because you miss a few days. Keep in mind that even a little meditation can help improve your memory. Therefore, try to meditate when you can, even if you don't find time to do it every day.\n14. **Try a guided meditation.** If you don't want to take a class, you can still benefit from the wisdom of others. Try doing a guided meditation. You can find many online, or you can download free apps. The person on the other end will walk you through a meditation process, helping you to learn how to do it.\n15. **Change it up.** You don't have to meditate the same way every time. For instance, some people find a walking meditation helpful. Take a ten-minute walk, focusing on different sensations in turn. Start with feeling your body walking, really focusing on what the movements feel like. Move on to the feeling of breathing. After that, focus on what the air feels like on your skin, then try thinking about just what you see and then just what you hear."
                ],
            },
            {
                "question": "How to Clean a Bathroom Fan",
                "answers": [
                    "One of the most neglected spaces when it comes to cleaning a bathroom is the fan. Having a clean, functional fan can lessen bathroom odors, as well as combat mold and mildew growth. These issues can become a health hazard if left unattended for too long. By cleaning your fan around every 6 months, you will be able to remove built up dirt before it becomes a problem.\n1. **Turn off the power.** Before you do anything else, ensure that the fan is turned off and cannot turn back on until you are finished cleaning it. Most models will have a plug that is located directly behind the cover. You could remove the cover first and unplug the fan, but just to be safe, go and temporarily pull the breaker for your bathroom. The fan is now safe to work on.\n2. **Remove the cover.** Dust will fall when the cover is removed. To avoid the dust, position your stepladder such that you can reach the cover, but are not standing directly below it. Most covers will have 2 prongs on opposite sides holding it in place, others just need to be unscrewed. Remove the cover by pressing these prongs in or removing the screws, then set the cover aside.\n3. **Remove the fan.** Unscrew the assembly that is holding the fan in place, then very gently remove the fan. Be careful not to drop the fan or hit it on the side of the exhaust pipe as that could potentially chip the fan blades. Broken fan blades will cause the fan to be louder and less effective.\n4. **Clean the cover and fan.** Start by vacuuming off the majority of the built up grime on both the cover and the fan. Then dip a rag, preferably a microfiber cloth, in soapy water and use it to wipe up the remaining dust. Be as thorough as you can, you will probably not do this again for a while.\nYou can let the cover soak in a tub of hot soapy water, but the fan should be wiped by hand to avoid getting water on the motor assembly or plug.\n5. **Vacuum the exhaust pipe.** Use a crevice or brush attachment and vacuum off the inside of the exhaust pipe. If you can reach, also use your rag or cloth to wipe off what the vacuum could not get.\n6. **Vacuum the external exhaust port.** This can be done later once the entire process is finished, but at some point you should go outside and find the exterior vent for your bathroom fan. Depending on where the bathroom is located, this vent will either be on the roof or the side of your house. Bring a damp rag to wipe off any dirt that has built up on the other end of your exhaust pipe.\n7. **Wipe and vacuum the fan housing.** If your fan had an accessible plug, be careful not to get any water inside the outlet. Doing so could result in electrocution or short circuit the fan when you plug it back in.  Therefore, use a dry rag to wipe off the fan housing, then vacuum up any remaining dust or debris.\n8. **Put the fan back in place.** Before reinstalling the fan, make sure that you cleaned off all the dust from in between each of the blades and dried it thoroughly. Carefully reinsert it into the exhaust pipe and screw the bracing back into place. Use your fingers and spin the fan around a few rotations to make sure that it is not rubbing against anything.\n9. **Turn the power back on.** Plug the fan back into the outlet and reset the breaker for your bathroom. The fan is now dangerous again, so do not touch it or continue to clean it after this point.\n10. **Reinstall the cover.** Once the cover has dried, either screw it back in or bend the prongs until the cover snaps back into place.\n11. **Test the fan.** Turn the fan on again to make sure everything works as normal. The fan should be quieter than it was before and provide a higher amount of air flow."
                ],
            },
            {
                "question": "How to Remove Smell from an Old Leather Bag",
                "answers": [
                    "Musty, stinky, odorous old leather bags aren't much fun and it's probable you're not keen to reuse such a bag. Before you resort to throwing it out, there are various ways that might just restore it to a respectable odor again.\n1. **Try a simple clean first.** If this clean doesn't shift the odor, you can try one of the other suggested methods after.\n\nWipe the leather bag inside and out with a clean, dry, soft cloth. This will pick up dust, loose debris and even some mold or mildew.\nWipe the leather bag down with a damp cloth. This will collect even more of the above items.\n2. **Allow the bag to air out.** Choose somewhere outdoors that is sheltered from direct light and heat, such as a table on the porch. Leave for a day if possible.\n3. **Check the odor.** If the bag still smells bad, choose one of the remaining suggested methods, or a combination of the methods.\n4. **Prepare a solution consisting of equal parts of white vinegar and distilled water.** Sponge the bag with the solution. Work on the inside of the bag and any mildewed outside part of the bag for a few minutes.\nIt's a good idea to test a small spot before trying this method, in case it stains.\n5. **Wipe off the vinegar solution with a clean, damp cloth.** \n6. **Allow to air dry.** Place the bag outside under shelter away from direct light to air dry.\n7. **Check the odor.** If it is still there, repeat. If not, the bag can be used again.\n8. **Use liquid detergent soap to clean the bag.** \n9. **Make a solution of soapy water, using the liquid detergent.** Dip the cleaning cloth or sponge in the solution and wring out before using.\n10. **Wipe the cloth over and inside the bag.** Concentrate in particular on the areas that you think are the smelliest.\n11. **Allow to air dry.** Place outside in a sheltered area away from direct sunlight and heat.\n12. **Once dry, check for the odor.** If it lingers, try again.\n13. **Use baking soda to deodorize the bag.** \n14. **Fill a clean sock with baking soda.** Tie off with a knot.\n15. **Place the leather bag and the baking soda-filled sock inside a large resealable plastic bag.** Alternatively, place both items inside an airtight container.\n16. **Set aside.** Let the baking soda work on the bag for at least 24 hours. The odors from the bag should transfer across to the baking soda.\n17. **Remove from the resealable bag or container.** Check the odor of the leather bag; if it still smells bad, repeat the process for another 24 hours, or longer. If it smells good again, throw away the baking soda, wash the sock and use the leather bag again.\n18. **Find some newspaper.** Scrunch the pages up and stuff them inside a large plastic bag, such as a kitchen waste bag or a garbage bag.\n19. **Slide the smelly leather bag in with the newspapers.** Arrange it so that it sits snugly in the middle of the papers.\n20. **Tie the bag up with a knot.** Alternatively, seal with a twist tie.\n21. **Let sit for at least 48 hours.** A few days more won't hurt it.\n22. **Remove from the bag.** Do a sniff test to see whether the odor has gone. If not, return to the bag for a few more days. Eventually it should start to smell better.\n23. **Fill a sock with coffee grounds.** They must be dry grounds, so if you're using grounds from your own coffee making, allow them to fully dry first. Or use the cheap instant coffee granules. Knot it off to keep the coffee intact.\n24. **Place the coffee sock inside your old leather bag.** Leave it there for up to a week. During this time, it should soak up much, if not all, of the cigarette smoke odor.\n25. **Do a smell test.** If all is good, the bag is ready for reuse. If it still smells a little, return the sock for a few more days.\n26. **Make or purchase some potpourri.** Place the potpourri inside a sachet.\n27. **Place the sachet inside the smelly bag.** Leave it there for at least one week.\n28. **Place the bag in an airy place.** Do not leave it in a dark cupboard; instead find somewhere with fresh air and indirect, cool light.\n29. **Check a week later.** It's a good idea to leave the sachet in the bag when using as well, as the scent will continue to improve the bag's own scent."
                ],
            },
            {
                "question": "How to Set up a Guppy Tank",
                "answers": ["Caring for guppies is relatively easy"],
            },
            {
                "question": "How to Fix Chemically Burnt Hair",
                "answers": [
                    "Many people suffer from hair that is damaged or burnt by various harsh chemical "
                ],
            },
        ]
        outputs = apply_metric(
            metric=metric,
            predictions=predictions,
            references=references,
            task_data=task_data,
        )
        actual_scores = [output["score"] for output in outputs]
        expected_scores = [
            {
                "global": {
                    "llm_as_judge": 0.1,
                    "score": 0.1,
                    "score_name": "llm_as_judge",
                },
                "instance": {
                    "llm_as_judge": 0.1,
                    "score": 0.1,
                    "score_name": "llm_as_judge",
                },
            },
            {
                "global": {
                    "llm_as_judge": 0.1,
                    "score": 0.1,
                    "score_name": "llm_as_judge",
                },
                "instance": {
                    "llm_as_judge": 0.1,
                    "score": 0.1,
                    "score_name": "llm_as_judge",
                },
            },
            {
                "global": {
                    "llm_as_judge": 0.1,
                    "score": 0.1,
                    "score_name": "llm_as_judge",
                },
                "instance": {
                    "llm_as_judge": 0.1,
                    "score": 0.1,
                    "score_name": "llm_as_judge",
                },
            },
            {
                "global": {
                    "llm_as_judge": 0.1,
                    "score": 0.1,
                    "score_name": "llm_as_judge",
                },
                "instance": {
                    "llm_as_judge": 0.1,
                    "score": 0.1,
                    "score_name": "llm_as_judge",
                },
            },
            {
                "global": {
                    "llm_as_judge": 0.1,
                    "score": 0.1,
                    "score_name": "llm_as_judge",
                },
                "instance": {
                    "llm_as_judge": 0.1,
                    "score": 0.1,
                    "score_name": "llm_as_judge",
                },
            },
        ]

        self.assertListEqual(actual_scores, expected_scores)
