from unitxt.metric_utils import (
    JoinSubsetsAndGroups,
    SplitSubsetsAndGroups,
)
from unitxt.stream import MultiStream

from tests.utils import UnitxtTestCase


class TestMetricUtils(UnitxtTestCase):
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
                        "groups": {
                            "mnli": {
                                "global": {
                                    "accuracy": 0.5,
                                    "exact_match": 1.0,
                                    "score": 0.5,
                                    "score_name": "accuracy",
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
                                            "accuracy": 0.0,
                                            "exact_match": 1.0,
                                            "score": 0.0,
                                            "score_name": "accuracy",
                                        }
                                    },
                                },
                                "groups": {
                                    "accuracy": 0.5,
                                    "exact_match": 1.0,
                                    "score": 0.5,
                                    "score_name": "accuracy",
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
                                            "accuracy": 0.0,
                                            "exact_match": 1.0,
                                            "score": 0.0,
                                            "score_name": "accuracy",
                                        }
                                    },
                                },
                                "score": 0.5,
                                "score_name": "accuracy",
                            },
                            "squad": {
                                "global": {
                                    "f1": 1.0,
                                    "exact_match": 1.0,
                                    "score": 1.0,
                                    "score_name": "f1",
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
                                "groups": {
                                    "f1": 1.0,
                                    "exact_match": 1.0,
                                    "score": 1.0,
                                    "score_name": "f1",
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
                                "score": 1.0,
                                "score_name": "f1",
                            },
                            "score": 0.75,
                            "score_name": "groups_mean",
                        },
                    },
                },
                {
                    "subset": ["mnli"],
                    "groups": ['{"template":"templates.t2"}', '{"num_demos": 1}'],
                    "score": {
                        "global": {
                            "accuracy": 0.5,
                            "exact_match": 1.0,
                            "score": 0.5,
                            "score_name": "accuracy",
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
                                    "accuracy": 0.0,
                                    "exact_match": 1.0,
                                    "score": 0.0,
                                    "score_name": "accuracy",
                                }
                            },
                        },
                        "instance": {
                            "accuracy": 0.0,
                            "exact_match": 1.0,
                            "score": 0.0,
                            "score_name": "accuracy",
                        },
                        "groups": {
                            "mnli": {
                                "global": {
                                    "accuracy": 0.5,
                                    "exact_match": 1.0,
                                    "score": 0.5,
                                    "score_name": "accuracy",
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
                                            "accuracy": 0.0,
                                            "exact_match": 1.0,
                                            "score": 0.0,
                                            "score_name": "accuracy",
                                        }
                                    },
                                },
                                "groups": {
                                    "accuracy": 0.5,
                                    "exact_match": 1.0,
                                    "score": 0.5,
                                    "score_name": "accuracy",
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
                                            "accuracy": 0.0,
                                            "exact_match": 1.0,
                                            "score": 0.0,
                                            "score_name": "accuracy",
                                        }
                                    },
                                },
                                "score": 0.5,
                                "score_name": "accuracy",
                            },
                            "squad": {
                                "global": {
                                    "f1": 1.0,
                                    "exact_match": 1.0,
                                    "score": 1.0,
                                    "score_name": "f1",
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
                                "groups": {
                                    "f1": 1.0,
                                    "exact_match": 1.0,
                                    "score": 1.0,
                                    "score_name": "f1",
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
                                "score": 1.0,
                                "score_name": "f1",
                            },
                            "score": 0.75,
                            "score_name": "groups_mean",
                        },
                    },
                },
                {
                    "subset": ["squad"],
                    "groups": ['{"template":"templates.t1"}', '{"num_demos": 1}'],
                    "score": {
                        "global": {
                            "f1": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "f1",
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
                        "instance": {
                            "f1": 1.0,
                            "exact_match": 1.0,
                            "score": 1.0,
                            "score_name": "f1",
                        },
                        "groups": {
                            "mnli": {
                                "global": {
                                    "accuracy": 0.5,
                                    "exact_match": 1.0,
                                    "score": 0.5,
                                    "score_name": "accuracy",
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
                                            "accuracy": 0.0,
                                            "exact_match": 1.0,
                                            "score": 0.0,
                                            "score_name": "accuracy",
                                        }
                                    },
                                },
                                "groups": {
                                    "accuracy": 0.5,
                                    "exact_match": 1.0,
                                    "score": 0.5,
                                    "score_name": "accuracy",
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
                                            "accuracy": 0.0,
                                            "exact_match": 1.0,
                                            "score": 0.0,
                                            "score_name": "accuracy",
                                        }
                                    },
                                },
                                "score": 0.5,
                                "score_name": "accuracy",
                            },
                            "squad": {
                                "global": {
                                    "f1": 1.0,
                                    "exact_match": 1.0,
                                    "score": 1.0,
                                    "score_name": "f1",
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
                                "groups": {
                                    "f1": 1.0,
                                    "exact_match": 1.0,
                                    "score": 1.0,
                                    "score_name": "f1",
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
                                "score": 1.0,
                                "score_name": "f1",
                            },
                            "score": 0.75,
                            "score_name": "groups_mean",
                        },
                    },
                },
            ],
        )
