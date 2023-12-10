<div align="center">
    <img src="./assets/banner.png" alt="Image Description" width="100%" />
</div>

Unitxt is a python library for getting data fired up and set for utilization.
In one line of code, it preps a dataset or mixtures-of-datasets into an input-output format for training and evaluation.
We aspire to be simple, adaptable and transparent.

Unitxt builds on separation. Separation allows adding a dataset, without knowing anything about the models using it. Separation allows training without caring for preprocessing, switching models without loading the data differently and changing formats (instruction\ICL\etc.) without changing anything else.

#
[![version](https://img.shields.io/pypi/v/unitxt)](https://pypi.org/project/unitxt/)
![license](https://img.shields.io/github/license/ibm/unitxt)
![python](https://img.shields.io/badge/python-3.8%20|%203.9-blue)
![tests](https://img.shields.io/github/actions/workflow/status/ibm/unitxt/tests.yml?branch=main&label=tests)
[![codecov](https://codecov.io/gh/IBM/unitxt/branch/main/graph/badge.svg?token=mlrWq9cwz3)](https://codecov.io/gh/IBM/unitxt)
![Read the Docs](https://img.shields.io/readthedocs/unitxt)
[![downloads](https://static.pepy.tech/personalized-badge/unitxt?period=total&units=international_system&left_color=grey&right_color=green&left_text=downloads)](https://pepy.tech/project/unitxt)

#

<div align="center">
    <img src="./assets/unitxt_flow_light.gif" alt="Unitxt Flow" width="100%" />
</div>

# Where to start? 🦄
[![Button](https://img.shields.io/badge/Overview-pink?style=for-the-badge)](https://unitxt.readthedocs.io/)
[![Button](https://img.shields.io/badge/Concepts-pink?style=for-the-badge)](https://unitxt.readthedocs.io/en/latest/concepts.html)
[![Button](https://img.shields.io/badge/Tutorial-pink?style=for-the-badge)](https://unitxt.readthedocs.io/)
[![Button](https://img.shields.io/badge/Examples-pink?style=for-the-badge)](https://unitxt.readthedocs.io/)
[![Button](https://img.shields.io/badge/Docs-pink?style=for-the-badge)](https://unitxt.readthedocs.io/)
# Why Unitxt? 🦄

### 🦄 Simplicity
Everything in Unitxt is simple and designed to feel natural and self-explanatory.
### 🦄 Adaptability
Adding new datasets, loading recipes, instructions and formatters is possible and encouraged!
### 🦄 Transparency
The resources and formatters of Unitxt are stored as shared datasets and therefore can easily reviewed by the crowd. Moreover, when assembling a dataset with Unitxt, it is very clear to others what's in it.

# Contributers

Please install Unitxt from source by:
```
git clone git@github.com:IBM/unitxt.git
cd unitxt
pip install -e ".[dev]"
pre-commit install
```
### Ensuring a Linear Git History

Configure your Git to maintain a linear history with these commands:

1. **Automatic Rebasing for Pulls**:
   - Command: `git config --global pull.rebase true`
   - This sets `git pull` to rebase changes, keeping your history linear without unnecessary merge commits.

2. **Fast-Forward Merges Only**:
   - Command: `git config --global merge.ff only`
   - This allows only fast-forward merges, preventing merge commits when branches diverge, to maintain a linear history.
