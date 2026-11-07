# CI workflows

## Why the runner image is pinned

Every job here sets `runs-on: ubuntu-24.04` rather than `ubuntu-latest`.

On 2026-11-05, GitHub moved the `ubuntu-latest` label to Ubuntu 26.04. That
broke five workflows at once, in two different ways:

1. **Python 3.8 and 3.9 are not available on the 26.04 image.** Jobs pinned to
   those versions failed during `actions/setup-python` with:

   ```
   The version '3.9' with architecture 'x64' was not found for Ubuntu 26.04.
   ```

   This hit `quality` (3.9), `docs` (3.8), `consistency` (3.9), `hub` (3.9) and
   `helm` (3.9). The authoritative list of which image ships which version is
   [actions/python-versions](https://raw.githubusercontent.com/actions/python-versions/main/versions-manifest.json).

2. **The 26.04 toolchain cannot build `pytrec-eval`.** Jobs on Python 3.10
   installed fine, then failed building the C extension, because a newer GCC
   rejects what the vendored `trec_eval` sources do:

   ```
   error: Failed to build `pytrec-eval==0.5`
   trec_eval-9.0.8/form_prefs_counts.c:818:12: error: passing argument 4 of
   'qsort' from incompatible pointer type [-Wincompatible-pointer-types]
   ```

   This hit `unittests`, `eager`, `performance`, `preparation` and the other
   3.10 jobs.

Note that `quality`'s failure output says the problem is "likely due to not
using pre-commit hooks". That message is misleading: the job prints it on any
failure, and in this case `setup-python` failed before pre-commit ever ran.

## Changing the pin

`pyproject.toml` declares `requires-python = ">=3.8"`, so CI deliberately still
tests 3.8 and 3.9. Moving the Python pins forward to satisfy a newer image
would quietly drop coverage for versions the package still claims to support —
pin the image instead, or drop the old versions from `requires-python` as a
separate, deliberate decision.

Before moving to a newer image, check that:

- the image ships every Python version these workflows pin, and
- `pytrec-eval` builds on it (or the dependency has been updated or dropped).
