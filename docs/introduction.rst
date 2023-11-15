.. image:: ../assets/banner.png
   :alt: Optional alt text
   :width: 100%
   :align: center

===================
Introduction
===================

Unitxt is a Python library for getting data ready for utilization in training, evaluation and inference of language models.
It provides a set of reusable building blocks and methodology for defining datasets and metrics.

In one line of code, it prepares a dataset or mixtures-of-datasets into an sequence to sequence input-output format for training and evaluation. Our aspiration is to be simple, adaptable, and transparent.

Unitxt builds on the principle of modularity. Modularity allows adding a dataset without knowing anything about the models using it. It allows for training without worrying about preprocessing, switching models without the need to load the data differently, and changing formats (instruction, in-context learning, etc.) without changing anything else.

Unitxt comes with a library of prefined task definitions, datasets, templates and metrics that cover most NLP tasks, including classification, extraction, summarization, question answering, and more.
