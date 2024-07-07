# Contributing Guide

## Accessing the project
First clone the project:

```bash
git clone https://github.com/IBM/unitxt.git
```

Then, navigate to the project directory:

```bash
cd unitxt
```

## Setting up the project
First, create a virtual environment:

```bash
python -m venv unitxt-venv
```

Then, activate the virtual environment:

```bash
source unitxt-venv/bin/activate
```

Then, install the project:

```bash
pip install -e ".[dev]"
```

## Running pre-commit before committing

First, install the pre-commit hooks:

```bash
pre-commit install
```

To run pre-commit before committing:

```bash
pre-commit run --all-files
```

Or simply run:

```bash
make pre-commit
```

This will run the pre-commit hooks on all files.

The pre-commit hooks will:
1. Check for any linting errors
2. Check for any formatting errors
3. Check for any security vulnerabilities
4. Check for spelling errors
4. Verify you used relative imports inside src/ directory
5. Verify you used library imports outside src/ directory

## Running Tests

First, install the project with the test dependencies:

```bash
pip install -e ".[tests]"
```
To run a specific test:

```bash
python -m unittest tests.test_<module>
```

To run all the tests:

```bash
python -m unittest
```
Bef

# Repo principles:

## Git

## Legal

We have tried to make it as easy as possible to make contributions. This applies to how we handle the legal aspects of contribution. We use the same approach - the Developer's Certificate of Origin 1.1 (DCO) - that the Linux® Kernel community uses to manage code contributions.

We simply ask that when submitting a patch for review, the developer must include a sign-off statement in the commit message.

Here is an example Signed-off-by line, which indicates that the submitter accepts the DCO:

Signed-off-by: John Doe <john.doe@example.com>
You can include this automatically when you commit a change to your local git repository using the following command:

git commit -s

### Commit
Always commit with a [good commit message](https://cbea.ms/git-commit/) and sign off:

Example:

```bash
git commit -s
```

### Push
Push into a new branch and open a PR.

Example:

```bash
git push origin main:<my-new-branch-name>
```

### Merge your PR to main
Use squash and merge to merge your PR to main.


## Structure

### Layout
The layout of the repo is [src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)

