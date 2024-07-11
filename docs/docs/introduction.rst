.. image:: ../../assets/banner.png
   :alt: Optional alt text
   :width: 100%
   :align: center

===================
Introduction
===================

Unitxt is an innovative library for textual data preparation and evaluation of generative language models.

In the dynamic landscape of generative NLP, traditional text processing pipelines limit research flexibility and reproducibility, as they are tailored to specific dataset, task, and model combinations.
The escalating complexity, involving system prompts, model-specific formats, instructions, and more, calls for a shift to a structured, modular, and customizable solution.

Unitxt deconstructs the data preparations and evaluation flows into modular components, enabling easy customization and sharing between practitioners.

Key Capabilities:

- Built in support for a variety of NLP tasks, including ones not typically found in other frameworks, such as multi label classification, targeted sentiment analysis, entity and relation extraction, table understanding, and retrieval augmented generation.

- Support for changing templates and formats.

- Supports loading data from different datasources (e.g Local files, Huggingface, Cloud Storage, Kaggle )

- Large collection of metrics (including LLM as Judges)

- Compatible with Huggingface Dataset and Metric APIs and can be used without installation

- The same Unitxt data preparation pipeline can be used in evaluation and during inference in production systems

- Removes the requirement to run user python code in dataset processing - reducing security risks

Unitxt can be used in standalone code, and is also integrated into common libraries and evaluation frameworks such as
`HuggingFace`_, `Helm`_, `LM-eval-harness`_. 

To get started, can explore the Unitxt :ref:`catalog <demo>`, and then see how you can load a :ref:`dataset<loading_datasets>` and  :ref:`evaluate <evaluating_datasets>` it in a just a few lines of code.
Finally, you can then learn how to :ref:`add new datasets <adding_dataset>`.

Beyond being a tool, Unitxt is a community-driven platform, empowering users to build, share, and advance their pipelines collaboratively.

Join the Unitxt community at https://github.com/IBM/unitxt!

.. _Unitxt: https://github.com/IBM/unitxt
.. _HuggingFace: https://huggingface.co/
.. _LM-eval-harness: https://github.com/EleutherAI/lm-evaluation-harness
.. _Helm: https://github.com/stanford-crfm/helm
