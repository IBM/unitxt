.. _adding_benchmark:

.. note::

   To use this tutorial, you need to :ref:`install Unitxt <install_unitxt>`.

=================
Benchmarks ✨
=================

This guide will assist you in adding or using your new benchmark in Unitxt.

Unitxt helps define the data you want to include in your benchmark and aggregate any final score you consider important.

The first tool to use in creating a benchmark is the Unitxt  :ref:`recipe <recipe>`.

To find more information about recipes, and how to start refer to :ref:`adding dataset guide <adding_dataset>`.

Once you have constructed a list of recipes, you can fuse them to create a benchmark.

Let's say we want to create the GLUE benchmark.

We can utilize the following Unitxt :ref:`cards <data_task_card>`:

 - ``cards.cola``
 - ``cards.mnli``
 - ``cards.mrpc``
 - ``cards.qnli``
 - ``cards.qqp``
 - ``cards.rte``
 - ``cards.sst2``
 - ``cards.stsb``
 - ``cards.wnli``

We can compile them together using Unitxt Benchmark:

.. code-block:: python

    from unitxt.benchmark import Benchmark
    from unitxt.standard import DatasetRecipe

    benchmark = Benchmark(
        format="formats.user_agent",
        max_samples_per_subset=5,
        loader_limit=300,
        subsets={
            "cola": DatasetRecipe(card="cards.cola", template="templates.classification.multi_class.instruction"),
            "mnli": DatasetRecipe(card="cards.mnli", template="templates.classification.multi_class.relation.default"),
            "mrpc": DatasetRecipe(card="cards.mrpc", template="templates.classification.multi_class.relation.default"),
            "qnli": DatasetRecipe(card="cards.qnli", template="templates.classification.multi_class.relation.default"),
            "rte": DatasetRecipe(card="cards.rte", template="templates.classification.multi_class.relation.default"),
            "sst2": DatasetRecipe(card="cards.sst2", template="templates.classification.multi_class.title"),
            "stsb": DatasetRecipe(card="cards.stsb", template="templates.regression.two_texts.title"),
            "wnli": DatasetRecipe(card="cards.wnli", template="templates.classification.multi_class.relation.default"),
        },
    )

Next, you can evaluate this benchmark by:

.. code-block:: python

    dataset = list(benchmark()["test"])

    # Inference using Flan-T5 Base via Hugging Face API
    model = HFPipelineBasedInferenceEngine(
        model_name="google/flan-t5-base", max_new_tokens=32
    )

    predictions = model(dataset)
    results = evaluate(predictions=predictions, data=dataset)

    print(results.subsets_scores.summary)

The result will contain the score per subset as well as the final global result:

.. code-block:: python

    ...
    mnli:
        ...
        score (float):
            0.4
        score_name (str):
            f1_micro
       ...
    mrpc:
        ...
        score (float):
            0.6
        score_name (str):
            f1_micro
        ...
    score (float):
        0.521666065848072
    score_name (str):
        subsets_mean


Saving and Loading Benchmarks
++++++++++++++++++++++++++++++

As always in Unitxt, you can save your benchmark to the catalog with:

.. code-block:: python

    add_to_catalog(benchmark, "benchmarks.glue")

Others can then load it from the catalog and evaluate on your benchmark with:

.. code-block:: python

    from unitxt import load_dataset

    dataset = load_dataset("benchmarks.glue")

If they want to modify the format or any other parameter of the benchmark, they can easily do so by:

.. code-block:: python

    from unitxt import load_dataset

    dataset = load_dataset("benchmarks.glue[format=formats.llama3]")

Additional Options
++++++++++++++++++

If you want to explore different templates, you can do so by defining a list of templates within any recipe. For instance:

.. code-block:: python

    DatasetRecipe(
        card="cards.cola",
        template=[
            "templates.classification.multi_class.instruction",
            "templates.classification.multi_class.title"
        ],
        group_by=["template"]
    )

This configuration will also provide the score per template for this recipe. To explore more configurations and capabilities, see the :ref:`evaluation guide <evaluating_datasets>`.