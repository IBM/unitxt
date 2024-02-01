import json
import os
import unittest

from service.metrics.client_config import (
    UNITXT_REMOTE_METRICS,
    UNITXT_REMOTE_METRICS_ENDPOINT,
    get_metrics_client_config,
)


class TestMetricsServiceClientConfig(unittest.TestCase):
    def test_defined_remote_metrics(self):
        expected_remote_metrics = [
            "metrics.rag.context_relevance",
            "metrics.rag.bert_k_precision",
        ]
        expected_remote_metrics_endpoint = "http://127.0.0.1:8000/compute"
        os.environ[UNITXT_REMOTE_METRICS] = json.dumps(expected_remote_metrics)
        os.environ[UNITXT_REMOTE_METRICS_ENDPOINT] = expected_remote_metrics_endpoint

        remote_metrics, remote_metrics_endpoint = get_metrics_client_config()
        self.assertListEqual(remote_metrics, expected_remote_metrics)
        self.assertEqual(remote_metrics_endpoint, expected_remote_metrics_endpoint)

    def test_no_remote_metrics(self):
        if UNITXT_REMOTE_METRICS in os.environ:
            del os.environ[UNITXT_REMOTE_METRICS]
        if UNITXT_REMOTE_METRICS_ENDPOINT in os.environ:
            del os.environ[UNITXT_REMOTE_METRICS_ENDPOINT]

        remote_metrics, remote_metrics_endpoint = get_metrics_client_config()
        self.assertListEqual(remote_metrics, [])
        self.assertEqual(remote_metrics_endpoint, None)

    def test_missing_endpoint(self):
        expected_remote_metrics = [
            "metrics.rag.context_relevance",
            "metrics.rag.bert_k_precision",
        ]
        os.environ[UNITXT_REMOTE_METRICS] = json.dumps(expected_remote_metrics)
        if UNITXT_REMOTE_METRICS_ENDPOINT in os.environ:
            del os.environ[UNITXT_REMOTE_METRICS_ENDPOINT]

        with self.assertRaises(RuntimeError):
            get_metrics_client_config()

    def test_misconfigured_remote_metrics_as_dict(self):
        wrong_remote_metrics_as_dict = {"key": "metrics.rag.context_relevance"}
        os.environ[UNITXT_REMOTE_METRICS] = json.dumps(wrong_remote_metrics_as_dict)

        with self.assertRaises(RuntimeError):
            get_metrics_client_config()

    def test_misconfigured_remote_metrics_not_containing_strings(self):
        wrong_remote_metrics_containing_an_inner_list = [
            ["metrics.rag.context_relevance"]
        ]
        os.environ[UNITXT_REMOTE_METRICS] = json.dumps(
            wrong_remote_metrics_containing_an_inner_list
        )

        with self.assertRaises(RuntimeError):
            get_metrics_client_config()
