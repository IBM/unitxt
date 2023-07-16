=================
Adding Datasets 
=================

In this tutriol you will see how to add a dataset to a local private catalog
and to load it or share it with others. 

.. note::

   To use this tutorial, you need to install `unitxt` with the following command:

   .. code-block:: bash
   
      pip install unitxt

The dataset in this tutriol will be process 
with a predfined pipeline called The Common Recipe.


The Common Recipe
------------------

The common recipe is a recipe for processing data based on different factors unique to 
the dataset and task. The common recipe can prepare the data in different ways, for example
with instructions or demonstrations etc. The arguments of the dataset are given to the 
common recipe in order to build its pipeline trough something called a Task Card.

.. code-block:: python
    from unitxt.blocks import CommonRecipe
    from unitxt.load import load_dataset

    recipe = CommonRecipe(
        card='cards.wnli',
        num_demos=5, # The number of demonstrations for in context learning
        demos_pool_size=100, # the size of the demonstartion pool to sample the 5 demonstrations
        instruction_item=0, # Which instruction from ones availble in the card
    )

    dataset = load_dataset(recipe)

The Task Card
----------------

The task card holds all the information sepcific to the dataset needed for its preparation.


The different components of the card are:
    - loader: the loader is responsible for loading the dataset from a source
    - preprocess_steps: the preprocess steps are a list of unitxt operators that will be applied to the dataset
    - task: the task is a definition of the task that can be obtained from the dataset
    - templates: the templates are templates for turning the data into text-to-text format

for example for `wnli` the card is defined as follows:

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
                metrics=['accuracy'],
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

once the card is defined, it can be used to load the dataset as follows:

.. code-block:: python
    recipe = CommonRecipe(
        card=card, # The card defined above
        num_demos=5,
        demos_pool_size=100,
        instruction_item=0, 
    )

    dataset = load_dataset(recipe)

but it is recommended to save the card to a local catalog and load it 
as explained in the next section.

Adding the dataset to the catalog
----------------------------------

Once the card is defined, it can be saved to a local catalog as follows:

.. code-block:: python
    from unitxt.catalog import add_to_catalog

    add_to_catalog(card, 'cards.wnli') # will be saved to CATALOG_DIR/cards/wnli

Then the dataset can be loaded as follows:

.. code-block:: python
    from unitxt.load import load_dataset

    recipe = CommonRecipe(
        card='cards.wnli',
        num_demos=5,
        demos_pool_size=100,
        instruction_item=0, 
    )

    dataset = load_dataset(recipe)

or even simpler:

.. code-block:: python
    from datasets import load_dataset

    dataset = load_dataset('unitxt/data', 'card=cards.wnli,num_demos=5,demos_pool_size=100,instruction_item=0')

And the same results as before will be obtained.

