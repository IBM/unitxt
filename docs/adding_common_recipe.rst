=================
Adding Datasets
=================

In this tutorial, you will learn how to add a dataset to a local private catalog
and how to load or share it with others.

.. note::

   To use this tutorial, you need to install `unitxt` with the following command:

   .. code-block:: bash

      pip install unitxt

The Task Card
----------------

The task card contains all the dataset-specific information needed for its preparation.

The different components of the card are:
    - loader: the loader is responsible for loading the dataset from a source
    - preprocess_steps: these are a list of unitxt operators that will be applied to the dataset
    - task: this is a definition of the task that can be derived from the dataset
    - templates: these are templates for converting the data into a text-to-text format

For example, for `wnli` the card is defined as follows:

.. code-block:: python

    from src.unitxt.blocks import (
        LoadHF,
        SplitRandomMix,
        AddFields,
        TaskCard,
        NormalizeListFields,
        FormTask,
        TemplatesList,
        InputOutputTemplate,
        MapInstanceValues
    )

    card = TaskCard(
            loader=LoadHF(path='glue', name='wnli'),
            preprocess_steps=[
                SplitRandomMix({'train': 'train[95%]', 'validation': 'train[5%]', 'test': 'validation'}),
                MapInstanceValues(mappers={'label': {"0": 'entailment', "1": 'not_entailment'}}),
                AddFields(
                fields={
                    'choices': ['entailment', 'not_entailment'],
                }
                ),
                NormalizeListFields(
                    fields=['choices']
                ),
            ],
            task=FormTask(
                inputs=['choices', 'sentence1', 'sentence2'],
                outputs=['label'],
                metrics=['metrics.accuracy'],
            ),
            templates=TemplatesList([
                InputOutputTemplate(
                    input_format="""
                        Given this sentence: {sentence1}, classify if this sentence: {sentence2} is {choices}.
                    """.strip(),
                    output_format='{label}',
                ),
            ])
        )


.. note::

   Read more about the stream operators such as `LoadHF`, `SplitRandomMix`
   and `AddFields` in the :ref:`lib` unitxt section.



It is recommended to save the card to a local catalog and load it
as explained in the next section.

Adding the Dataset to the Catalog
----------------------------------

Once the card is defined, it can be saved to a local catalog as follows:

.. code-block:: python

    from unitxt import add_to_catalog

    add_to_catalog(card, 'cards.wnli') # will be saved to a default catalog
                                       # if you want to save to a different
                                       # directory use the catalog_path argument
.. note::

   By default, a new artifact will be added to a local catalog stored
   in the library directory. To use a different catalog,
   use the `catalog_path` argument.

   In order to load automatically from your new catalog remember to
   register your new catalog by `unitxt.register_catalog('my_catalog')`
   or by setting the `UNITXT_ARTIFACTORIES` environment variable to include your catalog.

Once the card is defined,a Common Recipe can be used to load the dataset.
The Common Recipe can prepare the data in different ways, for example,
through instructions or demonstrations.

.. code-block:: python

    from unitxt.blocks import StandardRecipe
    from unitxt import load_dataset

    recipe = StandardRecipe(
        card='cards.wnli',
        num_demos=3, # The number of demonstrations for in-context learning
        demos_pool_size=100 # The size of the demonstration pool from which to sample the 5 demonstrations
    )

    dataset = load_dataset(recipe)


Or even simpler:

.. code-block:: python

    from datasets import load_dataset

    dataset = load_dataset('unitxt/data', 'card=cards.wnli,num_demos=5,demos_pool_size=100,instruction_item=0')

And the same results as before will be obtained.

Sharing the Dataset
--------------------

Once the dataset is loaded, it can be shared with others by simply sharing the card file
with them to paste into their local catalog.
