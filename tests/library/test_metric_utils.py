from unitxt.fusion import FixedFusion
from unitxt.metric_utils import MultiStreamScoreMean
from unitxt.metrics import Rouge
from unitxt.operators import IterableSource, MergeStreams, SplitByNestedGroup

from tests.utils import UnitxtTestCase


class TestMetricUtils(UnitxtTestCase):
    def test_rougel_simple_avg_with_fuse_and_split(self):
        from numpy import nanmean

        metric = Rouge(
            rouge_types=["rougeL"],
            ci_scores=["rougeL"],
            score_names=["rougeL"],
            n_resamples=None,
        )
        references = [
            ["hello", "there"],
            ["general kenobi", "general yoda"],
            ["I sing", "singing in the rain"],
            ["As a cloud", "I wonder"],
            ["Tel Mond", "Aviv Tel"],
            ["no such zone", "return to sender"],
            ["my party it is", "I cry if I want to"],
            ["tell him right now", "I know something"],
        ]
        predictions = [
            "hello there",
            "general kenobi",
            "I am singing",
            "I wandered",
            "Tel Aviv",
            "no such number",
            "its my party",
            "tell him",
        ]

        grand_input = [
            {"references": reference, "prediction": prediction}
            for (reference, prediction) in zip(references, predictions)
        ]
        # make the above input instances -- a fusion of first half and second half of grand_input:
        ms_fused_two_halves = FixedFusion(
            origins={
                "originH1": IterableSource({"test": grand_input[:4]}),
                "originH2": IterableSource({"test": grand_input[4:]}),
            }
        )()
        # and split the fused halves back into two named halves:
        split_ms_fused_two_halves = SplitByNestedGroup(number_of_fusion_generations=1)(
            ms_fused_two_halves
        )
        self.assertSetEqual(
            {"test~originH1", "test~originH2"}, set(split_ms_fused_two_halves.keys())
        )
        # and score them split: each stream separately:
        split_halves_through_rouge = metric(split_ms_fused_two_halves)
        # finally smear the halves' scores
        mean_scored_split_ms = MultiStreamScoreMean()(split_halves_through_rouge)
        res = {}
        for split, stream in mean_scored_split_ms.items():
            res[split] = list(stream)
        self.assertDictEqual(
            {
                "score_name": "groups_mean",
                "score": 0.6214285714285714,
                "originH1": {
                    "rougeL": 0.6416666666666666,
                    "score": 0.6416666666666666,
                    "score_name": "rougeL",
                },
                "originH2": {
                    "rougeL": 0.6011904761904762,
                    "score": 0.6011904761904762,
                    "score_name": "rougeL",
                },
            },
            res["test~originH1"][0]["score"]["global"],
        )
        self.assertDictEqual(
            {
                "score_name": "groups_mean",
                "score": 0.6214285714285714,
                "originH1": {
                    "rougeL": 0.6416666666666666,
                    "score": 0.6416666666666666,
                    "score_name": "rougeL",
                },
                "originH2": {
                    "rougeL": 0.6011904761904762,
                    "score": 0.6011904761904762,
                    "score_name": "rougeL",
                },
            },
            res["test~originH2"][0]["score"]["global"],
        )

        # now conduct two fusion generations, starting from quarters:
        grand_input = [
            {"references": reference, "prediction": prediction}
            for (reference, prediction) in zip(references, predictions)
        ]

        ms_first_two_quarters = FixedFusion(
            origins={
                "originQ1": IterableSource({"test": grand_input[:2]}),
                "originQ2": IterableSource({"test": grand_input[2:4]}),
            },
        )()
        ms_last_two_quarters = FixedFusion(
            origins={
                "originQ3": IterableSource({"test": grand_input[4:6]}),
                "originQ4": IterableSource({"test": grand_input[6:]}),
            },
        )()
        # now fuse first_two_quarters with last_two_quarters
        ms_all_four_quarters = FixedFusion(
            origins={
                "originH1": IterableSource(ms_first_two_quarters),
                "originH2": IterableSource(ms_last_two_quarters),
            },
        )()
        self.assertEqual(
            1,
            len(ms_all_four_quarters),
            "although fused from 4 multistreams, still the resulting multistream consists of one split: test",
        )
        self.assertEqual(next(iter(ms_all_four_quarters.keys())), "test")

        # split by group, down to quarters, and score each quarter separately
        split_ms = SplitByNestedGroup(number_of_fusion_generations=2)(
            ms_all_four_quarters
        )
        scored_split_ms = metric(split_ms)
        # now smear, generating the grouped-nested, detailed global score
        mean_scored_split_ms = MultiStreamScoreMean()(scored_split_ms)
        res = {}
        for split, stream in mean_scored_split_ms.items():
            res[split] = list(stream)
        self.assertDictEqual(
            {
                "originH1": {
                    "originQ1": {
                        "rougeL": 0.8333333333333333,
                        "score": 0.8333333333333333,
                        "score_name": "rougeL",
                    },
                    "originQ2": {"rougeL": 0.45, "score": 0.45, "score_name": "rougeL"},
                    "score": 0.6416666666666666,
                    "score_name": "groups_mean",
                },
                "originH2": {
                    "originQ3": {
                        "rougeL": 0.5833333333333333,
                        "score": 0.5833333333333333,
                        "score_name": "rougeL",
                    },
                    "originQ4": {
                        "rougeL": 0.6190476190476191,
                        "score": 0.6190476190476191,
                        "score_name": "rougeL",
                    },
                    "score": 0.6011904761904762,
                    "score_name": "groups_mean",
                },
                "score_name": "groups_mean",
                "score": 0.6214285714285714,
            },
            res["test~originH1/originQ2"][0]["score"]["global"],
        )
        self.assertEqual("originH2/originQ3", res["test~originH2/originQ3"][0]["group"])
        self.assertAlmostEqual(
            nanmean(
                [
                    res["test~originH1/originQ2"][0]["score"]["global"]["originH1"][
                        "score"
                    ],
                    res["test~originH1/originQ2"][0]["score"]["global"]["originH2"][
                        "score"
                    ],
                ]
            ),
            res["test~originH1/originQ2"][0]["score"]["global"]["score"],
        )
        # and finally - merge all 4 streams into one stream, named 'all',
        # the original stream-names are maintained in each instance["origin"]
        merged_mean_scored = MergeStreams()(res)
        outputs = list(merged_mean_scored["all"])
        self.assertIn(
            outputs[0]["origin"],
            [
                "test~originH1/originQ1",
                "test~originH1/originQ2",
                "test~originH2/originQ3",
                "test~originH2/originQ4",
            ],
        )
