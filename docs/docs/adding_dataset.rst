.. _adding_dataset:

.. note::

   To use this tutorial, you need to :ref:`install unitxt <install_unitxt>`.

=================
Adding Datasets ✨
=================

This guide will assist you in adding or using your new dataset in unitxt.

The information needed for loading your data will be defined in  :class:`TaskCard <unitxt.card.TaskCard>` class:

.. code-block:: python

    card = TaskCard(
        # will be defined in the rest of this guide
    )

Loading The Raw Data
---------------------

To load data from an external source, use a loader.
For example, to load the `wmt16` translation dataset from the HuggingFace hub:

.. code-block:: python

    loader=LoadHF(path="wmt16", name="de-en"),

More loaders for different sources are available in the  :class:`loaders <unitxt.loaders>` section.

The Task
---------

Your data usually corresponds to a task like translation, sentiment classification, or summarization.
To ensure compatibility and processing into textual training examples, define your task schema:


.. code-block:: python

    task=FormTask(
        inputs=["text", "source_language", "target_language"], # str, str, str
        outputs=["translation"], # str
        metrics=["metrics.bleu"],
    ),

We have predefined several tasks in the catalog's :ref:`Tasks section <catalog.tasks>`.

If a cataloged task fits your use case, call it by name:

.. code-block:: python

    task='tasks.translation.directed',


The Preprocessing pipeline
---------------------------

The preprocessing pipeline consists of operations to prepare your data according to the task's schema.

For example, prepare the dataset for translation task:


.. code-block:: python

    ...
    preprocess_steps=[
        CopyFields( # copy the fields to prepare the fields required by the task schema
            field_to_field=[
                ["translation/en", "text"],
                ["translation/de", "translation"],
            ],
        ),
        AddFields( # add new fields required by the task schema
            fields={
                "source_language": "english",
                "target_language": "deutch",
            }
        ),
    ]

For more built-in operators read :class:`operators <unitxt.operators>`.

Most data can be normalized to the task schema using built-in operators, ensuring your data is processed with verified high-standard streaming code.

For custom operators, refer to the :ref:`adding operator guide <adding_operator>`.

The Template
----------------



Templates convert data points into a model-friendly textual form.
If using a predefined task, choose from the corresponding templates available in the catalog's :ref:`Templates section <catalog.templates>`.

.. note::

   Use the :ref:`comprehnisve guide on templates  <adding_template>` for more templates features.

Alternively define your custom templates:

.. code-block:: python

    ..
    templates=TemplatesList([
        InputOutputTemplate(
            input_format="Translate this sentence from {source_language} to {target_language}: {text}.",
            output_format='{translation}',
        ),
    ])

Testing your card
-------------------

Once your card is ready you can test it:

.. code-block:: python

        from unitxt.card import TaskCard
        from unitxt.loaders import LoadHF
        from unitxt.operators import CopyFields, AddFields
        from unitxt.test_utils.card import test_card

         card = TaskCard(
            loader=LoadHF(path="wmt16", name="de-en"),
            preprocess_steps=[
                CopyFields( # copy the fields to prepare the fields required by the task schema
                    field_to_field=[
                        ["translation/en", "text"],
                        ["translation/de", "translation"],
                    ],
                ),
                AddFields( # add new fields required by the task schema
                    fields={
                        "source_language": "english",
                        "target_language": "deutch",
                    }
                ),
            ],
            task="tasks.translation.directed",
            templates="templates.translation.directed.all"
        )

        test_card(card)


Adding to the catalog
-----------------------

Once your card is ready and tested you can add it to the catalog.


.. code-block:: python

    from unitxt import add_to_catalog

    add_to_catalog(card, 'cards.wmt.en_de')

In the same way you can save also your custom templates and tasks.

.. note::
   By default, a new artifact will be added to a local catalog stored
   in the library directory. To use a different catalog,
   use the `catalog_path` argument.

   In order to load automatically from your new catalog remember to
   register your new catalog by `unitxt.register_catalog('my_catalog')`
   or by setting the `UNITXT_ARTIFACTORIES` environment variable to include your catalog.


Putting it all together!
------------------------

Now everything is ready to use the data! we use standard ICL recipe to load it:

.. code-block:: python

    from unitxt.standard import StandardRecipe
    from unitxt import load_dataset

    recipe = StandardRecipe(
        card='cards.wmt.en_de',
        num_demos=3, # The number of demonstrations for in-context learning
        demos_pool_size=100 # The size of the demonstration pool from which to sample the 5 demonstrations
    )

    dataset = load_dataset(recipe)


Or even simpler with hugginface datasets:

.. code-block:: python

    from datasets import load_dataset

    dataset = load_dataset('unitxt/data', 'card=cards.wmt.en_de,num_demos=5,demos_pool_size=100,instruction_item=0')

And the same results as before will be obtained.

Sharing the Dataset
--------------------

Once the dataset is loaded, it can be shared with others by simply sharing the card file
with them to paste into their local catalog.
