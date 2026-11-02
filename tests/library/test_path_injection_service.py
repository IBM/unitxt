"""Regression tests for CodeQL py/path-injection alerts cs-12 .. cs-16 (#1991).

CodeQL traced one taint flow for all five alerts: the ``metric_name`` URL path
parameter of the metrics service (``src/unitxt/service/metrics/main.py``) reaching
``open()`` in ``Artifact.is_artifact_file`` and ``utils.load_json``, because the
artifact loader accepts filesystem paths as well as catalog names.

The remediation has two halves, tested here:

1. The service rejects anything that is not a catalog name, so the untrusted
   value never reaches the loader.
2. ``load_json`` no longer echoes file contents into its error, so a path that
   does reach it cannot be used to read a file (CWE-209).

The library's own support for loading an artifact from a file path is a
documented feature and is deliberately left working.
"""

import json
import os
import tempfile
import unittest

from unitxt.artifact import (
    Artifact,
    fetch_artifact,
    is_name_legal_for_catalog,
    separate_inside_and_outside_square_brackets,
)
from unitxt.utils import load_json


def _service_guard_accepts(metric_name):
    """Mirrors the check performed by the service's ``compute`` endpoint.

    The endpoint itself needs fastapi, which is an optional dependency, so the
    guard logic is exercised directly here.
    """
    requested_name, _ = separate_inside_and_outside_square_brackets(metric_name)
    return bool(is_name_legal_for_catalog(requested_name))


class TestServiceMetricNameGuard(unittest.TestCase):
    def test_rejects_absolute_paths(self):
        for name in ("/etc/passwd", "/tmp/secret.txt", "/var/run/secrets/token"):
            with self.subTest(name=name):
                self.assertFalse(_service_guard_accepts(name))

    def test_rejects_traversal(self):
        for name in (
            "../../etc/passwd",
            "../catalog/thing.json",
            "metrics.accuracy/../../etc/passwd",
        ):
            with self.subTest(name=name):
                self.assertFalse(_service_guard_accepts(name))

    def test_rejects_windows_style_paths(self):
        for name in (r"C:\windows\win.ini", r"..\..\secret.txt"):
            with self.subTest(name=name):
                self.assertFalse(_service_guard_accepts(name))

    def test_rejects_a_real_artifact_file_path(self):
        """Even a genuine artifact file must not be loadable by path remotely."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metric.json")
            with open(path, "w") as handle:
                json.dump({"__type__": "accuracy"}, handle)

            self.assertTrue(
                Artifact.is_artifact_file(path),
                "sanity: the file is a loadable artifact, so only the guard stops it",
            )
            self.assertFalse(_service_guard_accepts(path))

    def test_accepts_plain_catalog_names(self):
        for name in ("metrics.accuracy", "accuracy", "metrics.rouge"):
            with self.subTest(name=name):
                self.assertTrue(_service_guard_accepts(name))

    def test_accepts_catalog_names_with_overrides(self):
        """Bracketed overrides must keep working; only the name part is checked."""
        self.assertTrue(_service_guard_accepts("metrics.f1_macro[average=macro]"))


class TestServiceEndpointAppliesTheGuard(unittest.TestCase):
    """The guard above is only meaningful if ``compute`` actually performs it.

    The endpoint needs fastapi (an optional dependency), so rather than calling
    it, assert that its source wires the validator in before the artifact
    loader. This fails if the check is removed or moved after get_artifact.
    """

    def _compute_source(self):
        service_main = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "src",
            "unitxt",
            "service",
            "metrics",
            "main.py",
        )
        with open(service_main) as handle:
            source = handle.read()
        start = source.index("def compute(")
        return source[start : source.index("\n@app.", start)]

    def test_compute_validates_the_name_before_loading_the_artifact(self):
        body = self._compute_source()
        self.assertIn("is_name_legal_for_catalog", body)
        self.assertIn("HTTPException", body)
        self.assertLess(
            body.index("is_name_legal_for_catalog"),
            body.index("get_artifact"),
            "the name must be validated before it reaches the artifact loader",
        )

    def test_compute_rejects_with_a_client_error(self):
        self.assertIn("status_code=400", self._compute_source())


class TestLoadJsonDoesNotLeakFileContents(unittest.TestCase):
    def test_error_omits_file_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "secret.env")
            # Stands in for whatever the file happens to contain; it only needs
            # to be a distinctive, non-JSON string that must not be echoed back.
            canary = "CANARY-must-not-appear-in-the-error-message"
            with open(path, "w") as handle:
                handle.write(canary)

            with self.assertRaises(RuntimeError) as ctx:
                load_json(path)

            message = str(ctx.exception)
            self.assertNotIn(canary, message)

    def test_error_still_identifies_the_file_and_position(self):
        """The message must stay useful for debugging a broken catalog file."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "broken.json")
            with open(path, "w") as handle:
                handle.write('{"a": 1,,}')

            with self.assertRaises(RuntimeError) as ctx:
                load_json(path)

            message = str(ctx.exception)
            self.assertIn("broken.json", message)
            self.assertIn("line", message)


class TestArtifactFilePathLoadingStillWorks(unittest.TestCase):
    """The library feature CodeQL's sink sits on must not regress."""

    def test_fetch_artifact_still_loads_from_a_file_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "my_metric.json")
            with open(path, "w") as handle:
                json.dump({"__type__": "accuracy"}, handle)

            artifact, catalog = fetch_artifact(path)
            self.assertEqual(type(artifact).__name__, "Accuracy")
            self.assertIsNone(catalog)

    def test_fetch_artifact_still_loads_from_a_catalog_name(self):
        artifact, catalog = fetch_artifact("metrics.accuracy")
        self.assertIsNotNone(artifact)
        self.assertIsNotNone(catalog)

    def test_load_json_still_parses_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ok.json")
            with open(path, "w") as handle:
                json.dump({"__type__": "accuracy", "n": 1}, handle)

            self.assertEqual(load_json(path), {"__type__": "accuracy", "n": 1})


if __name__ == "__main__":
    unittest.main()
