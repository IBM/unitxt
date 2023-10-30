import unittest
from math import isnan

from src.unitxt.metrics import (
    F1,
    Accuracy,
    BertScore,
    F1Macro,
    F1MacroMultiLabel,
    F1Micro,
    F1MicroMultiLabel,
    Reward,
    Rouge,
    SentenceBert,
    Squad,
    TokenOverlap,
)
from src.unitxt.test_utils.metrics import apply_metric


class TestMetrics(unittest.TestCase):
    def test_accuracy(self):
        metric = Accuracy()

        predictions = ["A", "B", "C"]
        references = [["B"], ["A"], ["C"]]

        outputs = apply_metric(metric=metric, predictions=predictions, references=references)

        expected_global_result = {
            "accuracy": 1 / 3,
            "score": 1 / 3,
            "score_name": "accuracy",
        }

        global_result = outputs[0]["score"]["global"].copy()
        # Only check the keys that are expected, i.e. exist in expected_global_result
        global_result = {
            key: value
            for key, value in global_result.items() if key in expected_global_result
        }
        self.assertDictEqual(global_result, expected_global_result)

        instance_targets = [
            {"accuracy": 0.0, "score": 0.0, "score_name": "accuracy"},
            {"accuracy": 0.0, "score": 0.0, "score_name": "accuracy"},
            {"accuracy": 1.0, "score": 1.0, "score_name": "accuracy"},
        ]
        for output, target in zip(outputs, instance_targets):
            self.assertDictEqual(output["score"]["instance"], target)

    def test_squad(self):
        metric = Squad()
        predictions = ["1976", "Beyonce", "climate change"]
        references = [["1976"], ["Beyoncé and Bruno Mars"], ["climate change"]]
        instance_targets = [
            {"exact_match": 100.0, "f1": 100.0, "score": 100.0, "score_name": "f1"},
            {"exact_match": 0.0, "f1": 0.0, "score": 0.0, "score_name": "f1"},
            {"exact_match": 100.0, "f1": 100.0, "score": 100.0, "score_name": "f1"},
        ]
        global_target = 100 * 2 / 3
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)

        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        for output, target in zip(outputs, instance_targets):
            self.assertEqual(output["score"]["instance"], target)

    def test_f1_micro(self):
        metric = F1Micro()
        references = [["cat"], ["dog"], ["dog"], ["dog"], ["cat"], ["cat"]]
        predictions = ["cat", "cat", "dog", "dog", "cat", "cat"]
        # F1 micro is equal to accuracy in multi class setting  (5/6)
        global_target = 0.8333333
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        self.assertEqual("f1_micro", outputs[0]["score"]["global"]["score_name"])
        self.assertEqual("f1_micro", outputs[0]["score"]["instance"]["score_name"])

    def test_f1_macro(self):
        metric = F1Macro()
        references = [["cat"], ["dog"], ["dog"], ["dog"], ["cat"], ["cat"]]
        predictions = ["cat", "cat", "dog", "dog", "cat", "cat"]
        # recall class 'dog'  = 2/3  = 0.666        precision= 2/2 = 1    f1 = 0.8
        # recall class 'cat'  = 3/3  = 1            precision= 3/4 = 0.75 f1  =0.857142857143
        # macro f1 = (0.8+0.847)/2
        global_target = 0.82857142
        global_target_dog = 0.8
        global_target_cat = 0.857142857143

        outputs = apply_metric(metric=metric, predictions=predictions, references=references)
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        self.assertAlmostEqual(global_target_dog, outputs[0]["score"]["global"]["f1_dog"])
        self.assertAlmostEqual(global_target_cat, outputs[0]["score"]["global"]["f1_cat"])
        self.assertEqual("f1_macro", outputs[0]["score"]["global"]["score_name"])
        self.assertEqual("f1_macro", outputs[0]["score"]["instance"]["score_name"])

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

        outputs = apply_metric(metric=metric, predictions=predictions, references=references)
        self.assertAlmostEqual(global_target_dog, outputs[0]["score"]["global"]["f1_dog"])
        self.assertAlmostEqual(global_target_cat, outputs[0]["score"]["global"]["f1_cat"])
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        self.assertEqual("f1_macro", outputs[0]["score"]["global"]["score_name"])
        self.assertEqual("f1_macro", outputs[0]["score"]["instance"]["score_name"])

    def test_f1_macro_multilabel(self):
        metric = F1MacroMultiLabel()
        references = [[["cat", "dog"]], [["dog"]], [["dog"]], [["dog"]], [["cat"]], [["cat"]]]
        predictions = [["cat"], ["2"], ["cat", "dog"], ["dog"], ["cat"], ["cat"]]
        # recall class 'dog'  = 2/4  = 0.5          precision= 2/2 = 1       f1 = 0.666666666667
        # recall class 'cat'  = 3/3  = 1            precision= 3/4 = 0.75    f1 = 0.857142857143
        # macro f1 = 0.9
        global_target = 0.76190476
        global_target_dog = 0.666666666667
        global_target_cat = 0.857142857143

        outputs = apply_metric(metric=metric, predictions=predictions, references=references)
        self.assertAlmostEqual(global_target_dog, outputs[0]["score"]["global"]["f1_dog"])
        self.assertAlmostEqual(global_target_cat, outputs[0]["score"]["global"]["f1_cat"])
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        self.assertEqual("f1_macro", outputs[0]["score"]["global"]["score_name"])
        self.assertEqual("f1_macro", outputs[0]["score"]["instance"]["score_name"])

    def test_f1_micro_multilabel(self):
        metric = F1MicroMultiLabel()
        references = [[["cat", "dog"]], [["dog"]], [["dog"]], [["dog"]], [["cat"]], [["cat"]]]
        predictions = [["cat"], ["2"], ["cat", "dog"], ["dog"], ["cat"], ["cat"]]
        # cat     TP=3  FP=1  FN=0  TN=2
        # dog     TP=2  FP=0  FN=2  TN=2
        # total   TP=5  FP=1  FN=2  TN=4
        # precision = TP / (FP + TP) = 5 / 6 = 0.8333333333
        # recall = TP /( FN + TP) =  5 / 7 = 0.7142857
        global_target = 0.769230760933

        outputs = apply_metric(metric=metric, predictions=predictions, references=references)
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])

    def test_f1_macro_multilabel_with_nones(self):
        metric = F1MacroMultiLabel()

        references = [[["none"]]]
        predictions = [["none"]]
        global_target = float("nan")
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)
        self.assertTrue(isnan(outputs[0]["score"]["global"]["score"]))

        references = [[["none"]]]
        predictions = [["x", "y"]]
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)
        self.assertTrue(isnan(outputs[0]["score"]["global"]["score"]))

        references = [[["none"]]]
        predictions = [["none", "x", "y"]]
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)
        self.assertTrue(isnan(outputs[0]["score"]["global"]["score"]))

        references = [[["x"]], [["y"]]]
        predictions = [["x"], ["x"]]
        global_target = 0.33333333333
        # Recall(x) = 1.0 Precion(x) = 0.5   --> F1(x) = 0.66666
        # recall(y) = 0.0 Precision(x) = NAN --> F1(y) = 0
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])

        references = [[["none"]], [["x"]], [["y"]], [["none"]], [["none"]]]
        predictions = [["none"], ["x"], ["x"], ["none"], ["none"]]
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])

    def test_f1_micro_multilabel_with_nones(self):
        metric = F1MicroMultiLabel()
        references = [[["none"]]]
        predictions = [["cat", "dog"]]

        outputs = apply_metric(metric=metric, predictions=predictions, references=references)
        self.assertTrue(isnan(outputs[0]["score"]["global"]["score"]))

        references = [[["none"]]]
        predictions = [["none"]]
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)
        self.assertTrue(isnan(outputs[0]["score"]["global"]["score"]))

        references = [[["sad"]], [["sad"]]]
        predictions = [["dog", "fustrated"], ["sad"]]
        # TP = 1 FN = 1 FP=0 .. precision=100 recall=0.5
        # sad  TP=1  FP=1  FN=0  TN=1
        #
        # precision = TP / (FP + TP) = 1 / 2 = 0.5
        # recall = TP /( FN + TP) =  1 / 1 = 1

        global_target = 0.66666666
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])

        references = [[["none"]], [["sad"]]]
        predictions = [["dog", "fustrated"], ["sad"]]
        # precision = TP / (FP + TP) = 1 / 1 = 1
        # recall = TP /( FN + TP) =  1 / 1 = 1

        global_target = 1
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])

    def test_f1_multiple_use(self):
        metric = F1MacroMultiLabel()
        references = [[["cat", "dog"]]]
        predictions = [["cat"]]
        # recall class 'dog'  = 0/1 = 0             precision= 0/0 = 1       f1 = 0
        # recall class 'cat'  = 1/1 = 1             precision= 1/1 = 1       f1 = 1
        global_target = 0.5
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])
        references = [[["horse"]]]
        predictions = [["horse"]]
        global_target = 1
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])

    def test_rouge(self):
        metric = Rouge()
        references = [["hello", "there"], ["general kenobi", "general yoda"]]
        predictions = ["hello there", "general kenobi"]
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)
        global_target = 5 / 6
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"])

    def test_rouge_l(self):
        metric = Rouge(use_aggregator=False, rouge_types=["rougeL"])
        references = [["hello", "there"], ["general kenobi", "general yoda"]]
        predictions = ["hello there", "general kenobi"]
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)
        global_target = [2 / 3, 1.0]
        self.assertListEqual(global_target, outputs[0]["score"]["global"]["score"])

    def test_token_overlap(self):
        metric = TokenOverlap()
        predictions = ["hello there general dude", "foo bar foobar"]
        references = [["hello there general kenobi", "hello there!"], ["foo bar foobar", "foo bar"]]
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)
        global_targets = {"f1": 7 / 8, "precision": 7 / 8, "recall": 1}
        for target, value in global_targets.items():
            self.assertAlmostEqual(value, outputs[0]["score"]["global"][target])

    def test_bert_score(self):
        metric = BertScore(model_name="microsoft/deberta-xlarge-mnli")
        predictions = ["hello there general dude", "foo bar foobar"]
        references = [["hello there general kenobi", "hello there!"], ["foo bar foobar", "foo bar"]]
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)
        global_targets = {"f1": 0.89818, "precision": 0.92830, "recall": 0.92185}
        for target, value in global_targets.items():
            self.assertAlmostEqual(value, outputs[0]["score"]["global"][target], places=5)

    def test_sentence_bert(self):
        metric = SentenceBert(model_name="sentence-transformers/all-mpnet-base-v2")
        predictions = ["hello there general dude", "foo bar foobar"]
        references = [["hello there general kenobi", "hello there!"], ["foo bar foobar", "foo bar"]]
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)
        global_target = 0.85614
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"], places=5)

    def test_reward(self):
        metric = Reward(model_name="OpenAssistant/reward-model-deberta-v3-large-v2")
        predictions = ["hello there General Dude", "foo bar foobar"]
        references = [["How do you greet General Dude"], ["What is your name?"]]
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)
        global_target = 0.08608
        self.assertAlmostEqual(global_target, outputs[0]["score"]["global"]["score"], places=5)


class TestConfidenceIntervals(unittest.TestCase):

    def test_instance_metric_confidence_interval(self):
        """
        Test the calculation of confidence intervals
        for an instance metric (Accuracy is used as an instance of
        an InstanceMetric)
        """
        metric = Accuracy()

        predictions = ["A", "B", "C", "D", "E"] * 20  # 100 predictions
        references = [["B"], ["B"], ["C"], ["D"], ["E"]] * 20  # 80% are correct (4/5)

        outputs = apply_metric(metric=metric, predictions=predictions, references=references)

        expected_global_result = {
            'accuracy_ci_low': 0.71,
            'accuracy_ci_high': 0.87,
            'score_ci_low': 0.71,
            'score_ci_high': 0.87,
        }

        global_result = outputs[0]["score"]["global"].copy()
        print(global_result)
        # Only check the keys that are expected, i.e. exist in expected_global_result
        global_result = {
            key: value
            for key, value in global_result.items() if key in expected_global_result
        }
        self.assertDictEqual(global_result, expected_global_result)

