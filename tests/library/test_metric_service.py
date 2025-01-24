import json
import os

import httpretty
from unitxt.metric_utils import (
    UNITXT_REMOTE_METRICS,
    UNITXT_REMOTE_METRICS_ENDPOINT,
    get_remote_metrics_endpoint,
    get_remote_metrics_names,
)
from unitxt.metrics import RemoteMetric
from unitxt.test_utils.metrics import test_metric

from tests.utils import UnitxtTestCase


class TestMetricsServiceClientConfig(UnitxtTestCase):
    def test_defined_remote_metrics(self):
        expected_remote_metrics = [
            "metrics.rag.context_relevance",
            "metrics.rag.bert_k_precision",
        ]
        expected_remote_metrics_endpoint = "http://127.0.0.1:8000/compute"
        os.environ[UNITXT_REMOTE_METRICS] = json.dumps(expected_remote_metrics)
        os.environ[UNITXT_REMOTE_METRICS_ENDPOINT] = expected_remote_metrics_endpoint

        remote_metrics = get_remote_metrics_names()
        self.assertListEqual(remote_metrics, expected_remote_metrics)

        remote_metrics_endpoint = get_remote_metrics_endpoint()
        self.assertEqual(remote_metrics_endpoint, expected_remote_metrics_endpoint)

    def test_no_remote_metrics(self):
        if UNITXT_REMOTE_METRICS in os.environ:
            del os.environ[UNITXT_REMOTE_METRICS]

        remote_metrics = get_remote_metrics_names()
        self.assertListEqual(remote_metrics, [])

    def test_missing_endpoint(self):
        if UNITXT_REMOTE_METRICS_ENDPOINT in os.environ:
            del os.environ[UNITXT_REMOTE_METRICS_ENDPOINT]

        with self.assertRaises(RuntimeError):
            get_remote_metrics_endpoint()

    def test_misconfigured_remote_metrics_as_dict(self):
        wrong_remote_metrics_as_dict = {"key": "metrics.rag.context_relevance"}
        os.environ[UNITXT_REMOTE_METRICS] = json.dumps(wrong_remote_metrics_as_dict)

        with self.assertRaises(RuntimeError):
            get_remote_metrics_names()

    def test_misconfigured_remote_metrics_not_containing_strings(self):
        wrong_remote_metrics_containing_an_inner_list = [
            ["metrics.rag.context_relevance"]
        ]
        os.environ[UNITXT_REMOTE_METRICS] = json.dumps(
            wrong_remote_metrics_containing_an_inner_list
        )

        with self.assertRaises(RuntimeError):
            get_remote_metrics_names()


class TestRemoteMetrics(UnitxtTestCase):
    @httpretty.activate(verbose=True)
    def test_remote_service_with_valid_response(self):
        """Test RemoteService with a mocked response that contains the expected response values."""
        instance_targets = [
            {
                "f1": 0.8,
                "precision": 0.86,
                "recall": 0.84,
                "score": 0.8,
                "score_name": "f1",
            },
            {
                "f1": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "score": 1.0,
                "score_name": "f1",
            },
        ]

        global_target = {
            "dummy_target_score_1": 0.9,
            "dummy_target_score_2": 1.0,
            "score_name": "dummy_target_score_1",
            "score": 0.9,
        }

        predictions = ["prediction text 1", "prediction text 2"]
        references = [
            ["reference text 1.1", "reference text 1.2"],
            ["reference text 2.1", "reference text 2.2"],
        ]

        def request_callback(request, uri, response_headers):
            content_type = request.headers.get("Content-Type")
            assert (
                content_type == "application/json"
            ), f"expected application/json but received Content-Type: {content_type}"

            # Check that the input is as expected
            inputs = json.loads(request.body)
            for instance_i, input_instance in enumerate(inputs["instance_inputs"]):
                self.assertEqual(input_instance["prediction"], predictions[instance_i])
                self.assertListEqual(
                    input_instance["references"], references[instance_i]
                )
                self.assertEqual(input_instance["additional_inputs"], {})

            # Return the expected response
            response = {
                "instances_scores": instance_targets,
                "global_score": global_target,
            }
            return [200, response_headers, json.dumps(response)]

        host = "www.dummy_hostname.com"
        endpoint = "http" + "://" + f"{host}/compute"
        metric_name = "metrics.bert_score.deberta.xlarge.mnli"
        httpretty.register_uri(
            httpretty.POST,
            f"{endpoint}/{metric_name}",
            body=request_callback,
        )

        metric = RemoteMetric(endpoint=endpoint, metric_name=metric_name)

        test_metric(
            metric=metric,
            predictions=predictions,
            references=references,
            instance_targets=instance_targets,
            global_target=global_target,
        )
