from unitxt.metric_utils import (
    JoinSubsetsAndGroups,
    SplitSubsetsAndGroups,
)
from unitxt.stream import MultiStream

from tests.utils import UnitxtTestCase


class TestMetricUtils(UnitxtTestCase):
    def test_split_none(self):
        operator = SplitSubsetsAndGroups()

        ms = MultiStream.from_iterables(
            {
                "test": [
                    {
                        "subset": [],
                        "groups": [],
                    },
                    {
                        "subset": [],
                        "groups": [],
                    },
                    {
                        "subset": [],
                        "groups": [],
                    },
                ]
            }
        )

        target = {
            "test://": [
                {
                    "subset": [],
                    "groups": [],
                    "__idx__": 0,
                },
                {
                    "subset": [],
                    "groups": [],
                    "__idx__": 1,
                },
                {
                    "subset": [],
                    "groups": [],
                    "__idx__": 2,
                },
            ],
        }

        result = operator(ms)

        self.assertEqual({k: list(v) for k, v in result.items()}, target)

    def test_split_groups(self):
        operator = SplitSubsetsAndGroups()

        ms = MultiStream.from_iterables(
            {
                "test": [
                    {
                        "subset": [],
                        "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    },
                    {
                        "subset": [],
                        "groups": ['{"template":"templates.t2"}', '{"num_demos": 1}'],
                    },
                    {
                        "subset": [],
                        "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    },
                ]
            }
        )

        target = {
            "test://": [
                {
                    "subset": [],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 0,
                },
                {
                    "subset": [],
                    "groups": ['{"template":"templates.t2"}', '{"num_demos": 1}'],
                    "__idx__": 1,
                },
                {
                    "subset": [],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 2,
                },
            ],
            "test://?template:templates.t1": [
                {
                    "subset": [],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 0,
                },
                {
                    "subset": [],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 2,
                },
            ],
            "test://?num_demos:1": [
                {
                    "subset": [],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 0,
                },
                {
                    "subset": [],
                    "groups": ['{"template":"templates.t2"}', '{"num_demos": 1}'],
                    "__idx__": 1,
                },
                {
                    "subset": [],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 2,
                },
            ],
            "test://?template:templates.t2": [
                {
                    "subset": [],
                    "groups": ['{"template":"templates.t2"}', '{"num_demos": 1}'],
                    "__idx__": 1,
                }
            ],
        }

        result = operator(ms)

        self.assertEqual({k: list(v) for k, v in result.items()}, target)

    def test_split_subsets(self):
        operator = SplitSubsetsAndGroups()

        ms = MultiStream.from_iterables(
            {
                "test": [
                    {
                        "subset": ["mnli"],
                        "groups": [],
                    },
                    {
                        "subset": ["mnli"],
                        "groups": [],
                    },
                    {
                        "subset": ["squad"],
                        "groups": [],
                    },
                ]
            }
        )

        target = {
            "test://mnli": [
                {
                    "subset": ["mnli"],
                    "groups": [],
                    "__idx__": 0,
                },
                {
                    "subset": ["mnli"],
                    "groups": [],
                    "__idx__": 1,
                },
            ],
            "test://squad": [
                {
                    "subset": ["squad"],
                    "groups": [],
                    "__idx__": 2,
                }
            ],
        }

        result = operator(ms)

        self.assertEqual({k: list(v) for k, v in result.items()}, target)

    def test_split_subset_and_groups(self):
        operator = SplitSubsetsAndGroups()

        ms = MultiStream.from_iterables(
            {
                "test": [
                    {
                        "subset": ["mnli"],
                        "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    },
                    {
                        "subset": ["mnli"],
                        "groups": ['{"template":"templates.t2"}', '{"num_demos": 1}'],
                    },
                    {
                        "subset": ["squad"],
                        "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    },
                ]
            }
        )

        target = {
            "test://mnli": [
                {
                    "subset": ["mnli"],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 0,
                },
                {
                    "subset": ["mnli"],
                    "groups": ['{"template":"templates.t2"}', '{"num_demos": 1}'],
                    "__idx__": 1,
                },
            ],
            "test://mnli?template:templates.t1": [
                {
                    "subset": ["mnli"],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 0,
                }
            ],
            "test://mnli?num_demos:1": [
                {
                    "subset": ["mnli"],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 0,
                },
                {
                    "subset": ["mnli"],
                    "groups": ['{"template":"templates.t2"}', '{"num_demos": 1}'],
                    "__idx__": 1,
                },
            ],
            "test://mnli?template:templates.t2": [
                {
                    "subset": ["mnli"],
                    "groups": ['{"template":"templates.t2"}', '{"num_demos": 1}'],
                    "__idx__": 1,
                }
            ],
            "test://squad": [
                {
                    "subset": ["squad"],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 2,
                }
            ],
            "test://squad?template:templates.t1": [
                {
                    "subset": ["squad"],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 2,
                }
            ],
            "test://squad?num_demos:1": [
                {
                    "subset": ["squad"],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 2,
                }
            ],
        }

        result = operator(ms)

        self.assertEqual({k: list(v) for k, v in result.items()}, target)

    def test_join_none(self):
        operator = JoinSubsetsAndGroups()
        inputs = {
            "test://": [
                {
                    "subset": [],
                    "groups": [],
                    "__idx__": 0,
                    "score": {
                        "global": {
                            "accuracy": 0.5,
                            "exact_match": 1.0,
                            "score": 0.5,
                            "score_name": "accuracy",
                        },
                        "instance": {
                            "accuracy": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "accuracy",
                        },
                    },
                },
                {
                    "subset": [],
                    "groups": [],
                    "__idx__": 1,
                    "score": {
                        "global": {
                            "accuracy": 0.5,
                            "exact_match": 1.0,
                            "score": 0.5,
                            "score_name": "accuracy",
                        },
                        "instance": {
                            "accuracy": 0.0,
                            "exact_match": 1.0,
                            "score": 0.0,
                            "score_name": "accuracy",
                        },
                    },
                },
            ],
        }

        ms = MultiStream.from_iterables(inputs)

        result = operator(ms)

        self.assertEqual(
            list(result["test"]),
            [
                {
                    "subset": [],
                    "groups": [],
                    "score": {
                        "instance": {
                            "accuracy": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "accuracy",
                        },
                        "global": {
                            "accuracy": 0.5,
                            "exact_match": 1.0,
                            "score": 0.5,
                            "score_name": "accuracy",
                        },
                    },
                },
                {
                    "subset": [],
                    "groups": [],
                    "score": {
                        "instance": {
                            "accuracy": 0.0,
                            "exact_match": 1.0,
                            "score": 0.0,
                            "score_name": "accuracy",
                        },
                        "global": {
                            "accuracy": 0.5,
                            "exact_match": 1.0,
                            "score": 0.5,
                            "score_name": "accuracy",
                        },
                    },
                },
            ],
        )

    def test_join_groups(self):
        operator = JoinSubsetsAndGroups()
        inputs = {
            "test://": [
                {
                    "subset": [],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 0,
                    "score": {
                        "global": {
                            "accuracy": 0.5,
                            "exact_match": 1.0,
                            "score": 0.5,
                            "score_name": "accuracy",
                        },
                        "instance": {
                            "accuracy": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "accuracy",
                        },
                    },
                },
                {
                    "subset": [],
                    "groups": ['{"template":"templates.t2"}', '{"num_demos": 1}'],
                    "__idx__": 1,
                    "score": {
                        "global": {
                            "accuracy": 0.5,
                            "exact_match": 1.0,
                            "score": 0.5,
                            "score_name": "accuracy",
                        },
                        "instance": {
                            "accuracy": 0.0,
                            "exact_match": 1.0,
                            "score": 0.0,
                            "score_name": "accuracy",
                        },
                    },
                },
                {
                    "subset": [],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 2,
                    "score": {
                        "global": {
                            "f1": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "f1",
                        },
                        "instance": {
                            "f1": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "f1",
                        },
                    },
                },
            ],
            "test://?template:templates.t1": [
                {
                    "subset": [],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 0,
                    "score": {
                        "global": {
                            "accuracy": 0.5,
                            "exact_match": 1.0,
                            "score": 0.5,
                            "score_name": "accuracy",
                        },
                        "instance": {
                            "accuracy": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "accuracy",
                        },
                    },
                },
                {
                    "subset": [],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 2,
                    "score": {
                        "global": {
                            "f1": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "f1",
                        },
                        "instance": {
                            "f1": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "f1",
                        },
                    },
                },
            ],
            "test://?num_demos:1": [
                {
                    "subset": [],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 0,
                    "score": {
                        "global": {
                            "accuracy": 0.5,
                            "exact_match": 1.0,
                            "score": 0.5,
                            "score_name": "accuracy",
                        },
                        "instance": {
                            "accuracy": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "accuracy",
                        },
                    },
                },
                {
                    "subset": [],
                    "groups": ['{"template":"templates.t2"}', '{"num_demos": 1}'],
                    "__idx__": 1,
                    "score": {
                        "global": {
                            "accuracy": 0.5,
                            "exact_match": 1.0,
                            "score": 0.5,
                            "score_name": "accuracy",
                        },
                        "instance": {
                            "accuracy": 0.0,
                            "exact_match": 1.0,
                            "score": 0.0,
                            "score_name": "accuracy",
                        },
                    },
                },
                {
                    "subset": [],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 2,
                    "score": {
                        "global": {
                            "f1": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "f1",
                        },
                        "instance": {
                            "f1": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "f1",
                        },
                    },
                },
            ],
            "test://?template:templates.t2": [
                {
                    "subset": [],
                    "groups": ['{"template":"templates.t2"}', '{"num_demos": 1}'],
                    "__idx__": 1,
                    "score": {
                        "global": {
                            "accuracy": 0.5,
                            "exact_match": 1.0,
                            "score": 0.5,
                            "score_name": "accuracy",
                        },
                        "instance": {
                            "accuracy": 0.0,
                            "exact_match": 1.0,
                            "score": 0.0,
                            "score_name": "accuracy",
                        },
                    },
                }
            ],
        }

        ms = MultiStream.from_iterables(inputs)

        result = operator(ms)

        self.assertEqual(
            list(result["test"]),
            [
                {
                    "subset": [],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "score": {
                        "instance": {
                            "accuracy": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "accuracy",
                        },
                        "global": {
                            "accuracy": 0.5,
                            "exact_match": 1.0,
                            "score": 0.5,
                            "score_name": "accuracy",
                        },
                        "groups": {
                            "template": {
                                "templates.t1": {
                                    "accuracy": 0.5,
                                    "exact_match": 1.0,
                                    "score": 0.5,
                                    "score_name": "accuracy",
                                },
                                "templates.t2": {
                                    "accuracy": 0.5,
                                    "exact_match": 1.0,
                                    "score": 0.5,
                                    "score_name": "accuracy",
                                },
                            },
                            "num_demos": {
                                "1": {
                                    "accuracy": 0.5,
                                    "exact_match": 1.0,
                                    "score": 0.5,
                                    "score_name": "accuracy",
                                }
                            },
                        },
                    },
                },
                {
                    "subset": [],
                    "groups": ['{"template":"templates.t2"}', '{"num_demos": 1}'],
                    "score": {
                        "instance": {
                            "accuracy": 0.0,
                            "exact_match": 1.0,
                            "score": 0.0,
                            "score_name": "accuracy",
                        },
                        "global": {
                            "accuracy": 0.5,
                            "exact_match": 1.0,
                            "score": 0.5,
                            "score_name": "accuracy",
                        },
                        "groups": {
                            "template": {
                                "templates.t1": {
                                    "accuracy": 0.5,
                                    "exact_match": 1.0,
                                    "score": 0.5,
                                    "score_name": "accuracy",
                                },
                                "templates.t2": {
                                    "accuracy": 0.5,
                                    "exact_match": 1.0,
                                    "score": 0.5,
                                    "score_name": "accuracy",
                                },
                            },
                            "num_demos": {
                                "1": {
                                    "accuracy": 0.5,
                                    "exact_match": 1.0,
                                    "score": 0.5,
                                    "score_name": "accuracy",
                                }
                            },
                        },
                    },
                },
                {
                    "subset": [],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "score": {
                        "instance": {
                            "f1": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "f1",
                        },
                        "global": {
                            "accuracy": 0.5,
                            "exact_match": 1.0,
                            "score": 0.5,
                            "score_name": "accuracy",
                        },
                        "groups": {
                            "template": {
                                "templates.t1": {
                                    "accuracy": 0.5,
                                    "exact_match": 1.0,
                                    "score": 0.5,
                                    "score_name": "accuracy",
                                },
                                "templates.t2": {
                                    "accuracy": 0.5,
                                    "exact_match": 1.0,
                                    "score": 0.5,
                                    "score_name": "accuracy",
                                },
                            },
                            "num_demos": {
                                "1": {
                                    "accuracy": 0.5,
                                    "exact_match": 1.0,
                                    "score": 0.5,
                                    "score_name": "accuracy",
                                }
                            },
                        },
                    },
                },
            ],
        )

    def test_join_subsets(self):
        operator = JoinSubsetsAndGroups()
        inputs = {
            "test://mnli": [
                {
                    "subset": ["mnli"],
                    "groups": [],
                    "__idx__": 0,
                    "score": {
                        "global": {
                            "accuracy": 0.5,
                            "exact_match": 1.0,
                            "score": 0.5,
                            "score_name": "accuracy",
                        },
                        "instance": {
                            "accuracy": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "accuracy",
                        },
                    },
                },
                {
                    "subset": ["mnli"],
                    "groups": [],
                    "__idx__": 1,
                    "score": {
                        "global": {
                            "accuracy": 0.5,
                            "exact_match": 1.0,
                            "score": 0.5,
                            "score_name": "accuracy",
                        },
                        "instance": {
                            "accuracy": 0.0,
                            "exact_match": 1.0,
                            "score": 0.0,
                            "score_name": "accuracy",
                        },
                    },
                },
            ],
            "test://squad": [
                {
                    "subset": ["squad"],
                    "groups": [],
                    "__idx__": 2,
                    "score": {
                        "global": {
                            "f1": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "f1",
                        },
                        "instance": {
                            "f1": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "f1",
                        },
                    },
                }
            ],
        }

        ms = MultiStream.from_iterables(inputs)

        result = operator(ms)

        self.assertEqual(
            list(result["test"]),
            [
                {
                    "subset": ["mnli"],
                    "groups": [],
                    "score": {
                        "instance": {
                            "accuracy": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "accuracy",
                        },
                        "subsets": {
                            "mnli": {
                                "accuracy": 0.5,
                                "exact_match": 1.0,
                                "score": 0.5,
                                "score_name": "accuracy",
                            },
                            "squad": {
                                "f1": 1.0,
                                "exact_match": 1.0,
                                "score": 1.0,
                                "score_name": "f1",
                            },
                            "score": 0.75,
                            "score_name": "subsets_mean",
                        },
                        "global": {"score": 0.75, "score_name": "subsets_mean"},
                    },
                },
                {
                    "subset": ["mnli"],
                    "groups": [],
                    "score": {
                        "instance": {
                            "accuracy": 0.0,
                            "exact_match": 1.0,
                            "score": 0.0,
                            "score_name": "accuracy",
                        },
                        "subsets": {
                            "mnli": {
                                "accuracy": 0.5,
                                "exact_match": 1.0,
                                "score": 0.5,
                                "score_name": "accuracy",
                            },
                            "squad": {
                                "f1": 1.0,
                                "exact_match": 1.0,
                                "score": 1.0,
                                "score_name": "f1",
                            },
                            "score": 0.75,
                            "score_name": "subsets_mean",
                        },
                        "global": {"score": 0.75, "score_name": "subsets_mean"},
                    },
                },
                {
                    "subset": ["squad"],
                    "groups": [],
                    "score": {
                        "instance": {
                            "f1": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "f1",
                        },
                        "subsets": {
                            "mnli": {
                                "accuracy": 0.5,
                                "exact_match": 1.0,
                                "score": 0.5,
                                "score_name": "accuracy",
                            },
                            "squad": {
                                "f1": 1.0,
                                "exact_match": 1.0,
                                "score": 1.0,
                                "score_name": "f1",
                            },
                            "score": 0.75,
                            "score_name": "subsets_mean",
                        },
                        "global": {"score": 0.75, "score_name": "subsets_mean"},
                    },
                },
            ],
        )

    def test_join_nested_subsets(self):
        operator = JoinSubsetsAndGroups()
        inputs = {
            "test://mnli/first": [
                {
                    "subset": ["mnli", "first"],
                    "groups": [],
                    "__idx__": 0,
                    "score": {
                        "global": {
                            "accuracy": 0.5,
                            "exact_match": 1.0,
                            "score": 0.5,
                            "score_name": "accuracy",
                        },
                        "instance": {
                            "accuracy": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "accuracy",
                        },
                    },
                },
                {
                    "subset": ["mnli", "first"],
                    "groups": [],
                    "__idx__": 1,
                    "score": {
                        "global": {
                            "accuracy": 0.5,
                            "exact_match": 1.0,
                            "score": 0.5,
                            "score_name": "accuracy",
                        },
                        "instance": {
                            "accuracy": 0.0,
                            "exact_match": 1.0,
                            "score": 0.0,
                            "score_name": "accuracy",
                        },
                    },
                },
            ],
            "test://mnli/second": [
                {
                    "subset": ["mnli", "second"],
                    "groups": [],
                    "__idx__": 0,
                    "score": {
                        "global": {
                            "accuracy": 0.5,
                            "exact_match": 1.0,
                            "score": 0.5,
                            "score_name": "accuracy",
                        },
                        "instance": {
                            "accuracy": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "accuracy",
                        },
                    },
                },
                {
                    "subset": ["mnli", "second"],
                    "groups": [],
                    "__idx__": 1,
                    "score": {
                        "global": {
                            "accuracy": 0.5,
                            "exact_match": 1.0,
                            "score": 0.5,
                            "score_name": "accuracy",
                        },
                        "instance": {
                            "accuracy": 0.0,
                            "exact_match": 1.0,
                            "score": 0.0,
                            "score_name": "accuracy",
                        },
                    },
                },
            ],
        }

        ms = MultiStream.from_iterables(inputs)

        result = operator(ms)

        self.assertEqual(
            list(result["test"]),
            [
                {
                    "subset": ["mnli", "first"],
                    "groups": [],
                    "score": {
                        "instance": {
                            "accuracy": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "accuracy",
                        },
                        "subsets": {
                            "mnli": {
                                "first": {
                                    "accuracy": 0.5,
                                    "exact_match": 1.0,
                                    "score": 0.5,
                                    "score_name": "accuracy",
                                },
                                "second": {
                                    "accuracy": 0.5,
                                    "exact_match": 1.0,
                                    "score": 0.5,
                                    "score_name": "accuracy",
                                },
                                "score": 0.5,
                                "score_name": "subsets_mean",
                            },
                            "score": 0.5,
                            "score_name": "subsets_mean",
                        },
                        "global": {"score": 0.5, "score_name": "subsets_mean"},
                    },
                },
                {
                    "subset": ["mnli", "first"],
                    "groups": [],
                    "score": {
                        "instance": {
                            "accuracy": 0.0,
                            "exact_match": 1.0,
                            "score": 0.0,
                            "score_name": "accuracy",
                        },
                        "subsets": {
                            "mnli": {
                                "first": {
                                    "accuracy": 0.5,
                                    "exact_match": 1.0,
                                    "score": 0.5,
                                    "score_name": "accuracy",
                                },
                                "second": {
                                    "accuracy": 0.5,
                                    "exact_match": 1.0,
                                    "score": 0.5,
                                    "score_name": "accuracy",
                                },
                                "score": 0.5,
                                "score_name": "subsets_mean",
                            },
                            "score": 0.5,
                            "score_name": "subsets_mean",
                        },
                        "global": {"score": 0.5, "score_name": "subsets_mean"},
                    },
                },
            ],
        )

    def test_join_subsets_and_groups(self):
        operator = JoinSubsetsAndGroups()

        inputs = {
            "test://mnli": [
                {
                    "subset": ["mnli"],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 0,
                    "score": {
                        "global": {
                            "accuracy": 0.5,
                            "exact_match": 1.0,
                            "score": 0.5,
                            "score_name": "accuracy",
                        },
                        "instance": {
                            "accuracy": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "accuracy",
                        },
                    },
                },
                {
                    "subset": ["mnli"],
                    "groups": ['{"template":"templates.t2"}', '{"num_demos": 1}'],
                    "__idx__": 1,
                    "score": {
                        "global": {
                            "accuracy": 0.5,
                            "exact_match": 1.0,
                            "score": 0.5,
                            "score_name": "accuracy",
                        },
                        "instance": {
                            "accuracy": 0.0,
                            "exact_match": 1.0,
                            "score": 0.0,
                            "score_name": "accuracy",
                        },
                    },
                },
            ],
            "test://mnli?template:templates.t1": [
                {
                    "subset": ["mnli"],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 0,
                    "score": {
                        "global": {
                            "accuracy": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "accuracy",
                        },
                        "instance": {
                            "accuracy": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "accuracy",
                        },
                    },
                }
            ],
            "test://mnli?num_demos:1": [
                {
                    "subset": ["mnli"],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 0,
                    "score": {
                        "global": {
                            "accuracy": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "accuracy",
                        },
                        "instance": {
                            "accuracy": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "accuracy",
                        },
                    },
                },
                {
                    "subset": ["mnli"],
                    "groups": ['{"template":"templates.t2"}', '{"num_demos": 1}'],
                    "__idx__": 1,
                    "score": {
                        "global": {
                            "accuracy": 0.0,
                            "exact_match": 1.0,
                            "score": 0.0,
                            "score_name": "accuracy",
                        },
                        "instance": {
                            "accuracy": 0.0,
                            "exact_match": 1.0,
                            "score": 0.0,
                            "score_name": "accuracy",
                        },
                    },
                },
            ],
            "test://mnli?template:templates.t2": [
                {
                    "subset": ["mnli"],
                    "groups": ['{"template":"templates.t2"}', '{"num_demos": 1}'],
                    "__idx__": 1,
                    "score": {
                        "global": {
                            "accuracy": 0.0,
                            "exact_match": 1.0,
                            "score": 0.0,
                            "score_name": "accuracy",
                        },
                        "instance": {
                            "accuracy": 0.0,
                            "exact_match": 1.0,
                            "score": 0.0,
                            "score_name": "accuracy",
                        },
                    },
                }
            ],
            "test://squad": [
                {
                    "subset": ["squad"],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 2,
                    "score": {
                        "global": {
                            "f1": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "f1",
                        },
                        "instance": {
                            "f1": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "f1",
                        },
                    },
                }
            ],
            "test://squad?template:templates.t1": [
                {
                    "subset": ["squad"],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 2,
                    "score": {
                        "global": {
                            "f1": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "f1",
                        },
                        "instance": {
                            "f1": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "f1",
                        },
                    },
                }
            ],
            "test://squad?num_demos:1": [
                {
                    "subset": ["squad"],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "__idx__": 2,
                    "score": {
                        "global": {
                            "f1": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "f1",
                        },
                        "instance": {
                            "f1": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "f1",
                        },
                    },
                }
            ],
        }

        ms = MultiStream.from_iterables(inputs)

        result = operator(ms)

        self.assertEqual(
            list(result["test"]),
            [
                {
                    "subset": ["mnli"],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "score": {
                        "instance": {
                            "accuracy": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "accuracy",
                        },
                        "subsets": {
                            "mnli": {
                                "accuracy": 0.5,
                                "exact_match": 1.0,
                                "score": 0.5,
                                "score_name": "accuracy",
                                "groups": {
                                    "template": {
                                        "templates.t1": {
                                            "accuracy": 1.0,
                                            "exact_match": 1.0,
                                            "score": 1.0,
                                            "score_name": "accuracy",
                                        },
                                        "templates.t2": {
                                            "accuracy": 0.0,
                                            "exact_match": 1.0,
                                            "score": 0.0,
                                            "score_name": "accuracy",
                                        },
                                    },
                                    "num_demos": {
                                        "1": {
                                            "accuracy": 1.0,
                                            "exact_match": 1.0,
                                            "score": 1.0,
                                            "score_name": "accuracy",
                                        }
                                    },
                                },
                            },
                            "squad": {
                                "f1": 1.0,
                                "exact_match": 1.0,
                                "score": 1.0,
                                "score_name": "f1",
                                "groups": {
                                    "template": {
                                        "templates.t1": {
                                            "f1": 1.0,
                                            "exact_match": 1.0,
                                            "score": 1.0,
                                            "score_name": "f1",
                                        }
                                    },
                                    "num_demos": {
                                        "1": {
                                            "f1": 1.0,
                                            "exact_match": 1.0,
                                            "score": 1.0,
                                            "score_name": "f1",
                                        }
                                    },
                                },
                            },
                            "score": 0.75,
                            "score_name": "subsets_mean",
                        },
                        "global": {"score": 0.75, "score_name": "subsets_mean"},
                    },
                },
                {
                    "subset": ["mnli"],
                    "groups": ['{"template":"templates.t2"}', '{"num_demos": 1}'],
                    "score": {
                        "instance": {
                            "accuracy": 0.0,
                            "exact_match": 1.0,
                            "score": 0.0,
                            "score_name": "accuracy",
                        },
                        "subsets": {
                            "mnli": {
                                "accuracy": 0.5,
                                "exact_match": 1.0,
                                "score": 0.5,
                                "score_name": "accuracy",
                                "groups": {
                                    "template": {
                                        "templates.t1": {
                                            "accuracy": 1.0,
                                            "exact_match": 1.0,
                                            "score": 1.0,
                                            "score_name": "accuracy",
                                        },
                                        "templates.t2": {
                                            "accuracy": 0.0,
                                            "exact_match": 1.0,
                                            "score": 0.0,
                                            "score_name": "accuracy",
                                        },
                                    },
                                    "num_demos": {
                                        "1": {
                                            "accuracy": 1.0,
                                            "exact_match": 1.0,
                                            "score": 1.0,
                                            "score_name": "accuracy",
                                        }
                                    },
                                },
                            },
                            "squad": {
                                "f1": 1.0,
                                "exact_match": 1.0,
                                "score": 1.0,
                                "score_name": "f1",
                                "groups": {
                                    "template": {
                                        "templates.t1": {
                                            "f1": 1.0,
                                            "exact_match": 1.0,
                                            "score": 1.0,
                                            "score_name": "f1",
                                        }
                                    },
                                    "num_demos": {
                                        "1": {
                                            "f1": 1.0,
                                            "exact_match": 1.0,
                                            "score": 1.0,
                                            "score_name": "f1",
                                        }
                                    },
                                },
                            },
                            "score": 0.75,
                            "score_name": "subsets_mean",
                        },
                        "global": {"score": 0.75, "score_name": "subsets_mean"},
                    },
                },
                {
                    "subset": ["squad"],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "score": {
                        "instance": {
                            "f1": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "f1",
                        },
                        "subsets": {
                            "mnli": {
                                "accuracy": 0.5,
                                "exact_match": 1.0,
                                "score": 0.5,
                                "score_name": "accuracy",
                                "groups": {
                                    "template": {
                                        "templates.t1": {
                                            "accuracy": 1.0,
                                            "exact_match": 1.0,
                                            "score": 1.0,
                                            "score_name": "accuracy",
                                        },
                                        "templates.t2": {
                                            "accuracy": 0.0,
                                            "exact_match": 1.0,
                                            "score": 0.0,
                                            "score_name": "accuracy",
                                        },
                                    },
                                    "num_demos": {
                                        "1": {
                                            "accuracy": 1.0,
                                            "exact_match": 1.0,
                                            "score": 1.0,
                                            "score_name": "accuracy",
                                        }
                                    },
                                },
                            },
                            "squad": {
                                "f1": 1.0,
                                "exact_match": 1.0,
                                "score": 1.0,
                                "score_name": "f1",
                                "groups": {
                                    "template": {
                                        "templates.t1": {
                                            "f1": 1.0,
                                            "exact_match": 1.0,
                                            "score": 1.0,
                                            "score_name": "f1",
                                        }
                                    },
                                    "num_demos": {
                                        "1": {
                                            "f1": 1.0,
                                            "exact_match": 1.0,
                                            "score": 1.0,
                                            "score_name": "f1",
                                        }
                                    },
                                },
                            },
                            "score": 0.75,
                            "score_name": "subsets_mean",
                        },
                        "global": {"score": 0.75, "score_name": "subsets_mean"},
                    },
                },
            ],
        )
