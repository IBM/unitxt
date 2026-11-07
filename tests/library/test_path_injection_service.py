"""Regression tests for CodeQL py/path-injection alerts cs-12..cs-18 (#1991, #1994).

CodeQL traced one taint flow for every alert: the ``metric_name`` URL path
parameter of the metrics service (``src/unitxt/service/metrics/main.py``)
reaching ``open()`` in ``Artifact.is_artifact_file`` and ``utils.load_json``,
because the artifact loader accepts filesystem paths as well as catalog names.

Two earlier attempts did not clear the alerts, and both were wrong for the same
underlying reason -- the value handed to the loader was still derived from the
request:

- #1993 validated a derived value and then passed the original ``metric_name``
  on, so the check said nothing about what actually reached the loader.
- #1995 returned a name rebuilt from the validated parts. That reads safer, but
  ``split``/``join`` still derive the result from the input, so it was no safer
  than the checks applied to it and the taint path survived intact.

The remediation tested here resolves the request against the catalog and returns
the catalog's own string, so the value passed on originates in trusted data
rather than in the request:

1. ``sanitize_catalog_identifier`` looks the name up and returns the catalog's
   copy of it, rejecting anything the catalogs do not publish.
2. The service uses only that returned name, and answers 400 otherwise.
3. ``load_json`` does not echo file contents into its error, so a path that does
   reach it cannot be used to read a file (CWE-209).

The library's own support for loading an artifact from a file path is a
documented feature and is deliberately left working.
"""

import json
import os
import tempfile
import unittest

from unitxt import artifact as artifact_module
from unitxt.artifact import (
    Artifact,
    IllegalCatalogNameError,
    _catalog_name_index,
    fetch_artifact,
    sanitize_catalog_identifier,
)
from unitxt.operators import ArtifactFetcherMixin
from unitxt.utils import load_json

# Shapes that must never be accepted from an untrusted caller. Each would
# otherwise be opened as a filesystem path by the artifact loader.
UNSAFE_REFERENCES = [
    "/etc/passwd",
    "/tmp/secret.txt",
    "/var/run/secrets/token",
    "../../etc/passwd",
    "../catalog/thing.json",
    "metrics.accuracy/../../etc/passwd",
    r"C:\windows\win.ini",
    r"..\..\secret.txt",
    "~/.ssh/id_rsa",
    "./relative.json",
    "/",
    "",
    "metrics\x00.accuracy",
    "http://evil.example/x.json",
]


class TestSanitizeCatalogIdentifierRejectsPaths(unittest.TestCase):
    def test_rejects_unsafe_references(self):
        for name in UNSAFE_REFERENCES:
            with self.subTest(name=name):
                with self.assertRaises(IllegalCatalogNameError):
                    sanitize_catalog_identifier(name)

    def test_rejects_malformed_brackets(self):
        """Must be a clean rejection, not an unhandled ValueError from the parser."""
        for name in ("metrics.accuracy[unclosed", "a]b[c", "metrics.x]"):
            with self.subTest(name=name):
                with self.assertRaises(IllegalCatalogNameError):
                    sanitize_catalog_identifier(name)

    def test_rejects_non_strings(self):
        for value in (None, 123, ["metrics.accuracy"], {"a": 1}):
            with self.subTest(value=value):
                with self.assertRaises(IllegalCatalogNameError):
                    sanitize_catalog_identifier(value)

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
            with self.assertRaises(IllegalCatalogNameError):
                sanitize_catalog_identifier(path)

    def test_error_names_the_offending_value(self):
        with self.assertRaises(IllegalCatalogNameError) as ctx:
            sanitize_catalog_identifier("/etc/passwd")
        self.assertEqual(ctx.exception.name, "/etc/passwd")

    def test_is_a_value_error(self):
        """Callers catching ValueError must keep working."""
        self.assertTrue(issubclass(IllegalCatalogNameError, ValueError))


class TestSanitizeCatalogIdentifierAcceptsNames(unittest.TestCase):
    def test_accepts_plain_catalog_names(self):
        # Must be names the catalogs actually publish; a bare "accuracy" is not
        # one (it does not resolve on an unmodified tree either).
        for name in ("metrics.accuracy", "metrics.rouge", "metrics.f1_macro"):
            with self.subTest(name=name):
                self.assertEqual(sanitize_catalog_identifier(name), (name, None))

    def test_accepts_catalog_names_with_overrides(self):
        """Bracketed overrides must keep working; only the name part is a path risk."""
        name, overrides = sanitize_catalog_identifier(
            "metrics.f1_macro[ignore_case=False]"
        )
        self.assertEqual(name, "metrics.f1_macro")
        self.assertEqual(overrides, {"ignore_case": False})

    def test_returns_the_catalogs_own_string_not_the_callers(self):
        """The point of the fix: the returned value comes from the catalog.

        Rebuilding the name from the caller's input (an earlier attempt) left a
        value still derived from the request. Resolving it against the catalog
        replaces it with a string that originates in the catalog listing, which
        is what lets callers pass it on without re-validating. Asserting
        identity, not equality, is what distinguishes the two.
        """
        requested = "".join(["metrics", ".", "accuracy"])
        returned, _ = sanitize_catalog_identifier(requested)

        self.assertEqual(returned, requested)
        self.assertIsNot(
            returned,
            requested,
            "the returned name must not be the caller's own string object",
        )
        index = _catalog_name_index(artifact_module._catalog_name_generation)
        self.assertIs(
            returned,
            index["metrics.accuracy"],
            "the returned name must be the catalog's own string object",
        )

    def test_strips_overrides_from_the_returned_name(self):
        returned, overrides = sanitize_catalog_identifier(
            "metrics.f1_macro[ignore_case=False]"
        )
        self.assertNotIn("[", returned)
        self.assertEqual(returned, "metrics.f1_macro")
        self.assertEqual(overrides, {"ignore_case": False})

    def test_only_local_catalog_names_are_indexed(self):
        """Documents a deliberate limitation, so a change here is a decision.

        The index is built by walking local catalogs, so a name published only
        by a remote catalog is not resolvable and the service rejects it. See
        the note on ``_catalog_name_index``.
        """
        index = _catalog_name_index(artifact_module._catalog_name_generation)
        from unitxt.catalog import get_local_catalogs_paths, ls

        self.assertEqual(
            set(index), set(ls(to_file=os.devnull)), "index must match the local walk"
        )
        self.assertTrue(
            get_local_catalogs_paths(), "sanity: there is at least one local catalog"
        )

    def test_rejects_names_the_catalogs_do_not_publish(self):
        """A syntactically legal name that does not exist must not reach a loader."""
        for name in (
            "metrics.no_such_metric_zzz",
            "not_a_real_namespace.thing",
        ):
            with self.subTest(name=name):
                with self.assertRaises(IllegalCatalogNameError):
                    sanitize_catalog_identifier(name)


class TestSanitizedNameIsWhatTheServiceUses(unittest.TestCase):
    """The sanitizer only helps if ``compute`` passes on its result.

    The endpoint needs fastapi (an optional dependency), so rather than calling
    it, assert its source uses the sanitized name and never the raw parameter.
    This fails if the two are edited apart again, which is what kept the alerts
    open after the first fix.
    """

    @staticmethod
    def _compute_source():
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

    def test_compute_sanitizes_before_loading_the_artifact(self):
        body = self._compute_source()
        self.assertIn("sanitize_catalog_identifier", body)
        self.assertLess(
            body.index("sanitize_catalog_identifier"),
            body.index("get_artifact"),
            "the name must be sanitized before it reaches the artifact loader",
        )

    def test_compute_rejects_with_a_client_error(self):
        body = self._compute_source()
        self.assertIn("IllegalCatalogNameError", body)
        self.assertIn("status_code=400", body)

    def test_compute_never_passes_the_raw_parameter_onward(self):
        """``metric_name`` must be consumed by the sanitizer and nothing else."""
        body = self._compute_source()
        uses = [
            line.strip()
            for line in body.splitlines()
            if "metric_name" in line
            and "safe_metric_name" not in line
            and not line.strip().startswith("#")
        ]
        # Only the signature and the sanitizer call may mention the raw value.
        for line in uses:
            with self.subTest(line=line):
                self.assertTrue(
                    "metric_name: str" in line
                    or "sanitize_catalog_identifier(metric_name)" in line,
                    f"raw metric_name must not be used here: {line}",
                )


class TestGetArtifactOverrides(unittest.TestCase):
    """Overrides now travel beside the name, so they must still apply."""

    def test_overrides_are_applied(self):
        name, overrides = sanitize_catalog_identifier(
            "metrics.f1_macro[ignore_case=False]"
        )
        artifact = ArtifactFetcherMixin.get_artifact(name, overwrite_kwargs=overrides)
        self.assertFalse(artifact.ignore_case)

    def test_overrides_do_not_poison_the_cache_for_the_plain_artifact(self):
        plain_before = ArtifactFetcherMixin.get_artifact("metrics.f1_macro")
        self.assertTrue(plain_before.ignore_case, "sanity: default is True")

        name, overrides = sanitize_catalog_identifier(
            "metrics.f1_macro[ignore_case=False]"
        )
        overridden = ArtifactFetcherMixin.get_artifact(name, overwrite_kwargs=overrides)
        self.assertFalse(overridden.ignore_case)

        plain_after = ArtifactFetcherMixin.get_artifact("metrics.f1_macro")
        self.assertTrue(
            plain_after.ignore_case,
            "an overridden fetch must not be served from, or written to, "
            "the cache entry of the plain artifact",
        )

    def test_without_overrides_behaves_as_before(self):
        artifact = ArtifactFetcherMixin.get_artifact("metrics.accuracy")
        self.assertIsNotNone(artifact)


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
