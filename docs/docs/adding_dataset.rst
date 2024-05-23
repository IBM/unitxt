.. _adding_dataset:

.. note::

   To use this tutorial, you need to :ref:`install unitxt <install_unitxt>`.

=================
Datasets ✨
=================

This guide will assist you in adding or using your new dataset in unitxt.

The information needed for loading your data will be defined in  :class:`TaskCard <unitxt.card.TaskCard>` class:

.. code-block:: python

    card = TaskCard(
        # will be defined in the rest of this guide
    )


The Task
---------

When we "add a dataset", we are actually adding data for a particiular NLP task such as translation, sentiment classification, question answering, summarization, etc.
In fact, the same dataset can be used for multiple NLP tasks. For example, a dataset with question and answer pairs, can be used for both
question answering and question generation.  Similarly, a dataset with corresponding English and French sentences, can be used for
an Engish to French translation task or for a French to English translation task.

The Task schema is a formal definition of the NLP task , including its inputs, outputs, and default evaluation metrics.

The `inputs` of the task are a set of fields that are used to format the textual input to the model.
The `output` of the task are a set of fields that are used to format the textual expected output from the model (gold references).
The `metrics` of the task are a set of default metrics to be used to evaluate the outputs of the model.

While language models generate textual predictions, the metrics often evaluate on a different datatypes.  For example,
spearman correlation is evaluated on numeric predictions vs numeric reference, and multi-label F1 is evaluated on a list of string class name prediction_type
vs a reference list of string classes.  The `prediction_type` of the task defines the common prediction (and reference) types for all metrics of the task.

Note that the the Task does not perform any verbalization or formatting of the task input and output fields - this is the responsibility of the Template.

In our example, we will formalize a translation task between `source_language` and a `target_language`.
The text to translate is in the field `text` and the reference answer in the `translation` field.
We will use the `bleu` metric for a reference based evaluation.

.. code-block:: python

    task=Task(
        inputs= { "text" : "str", "source_language" : "str", "target_language" : "str"},
        outputs= {"translation" : "str"},
        prediction_type="str",
        metrics=["metrics.bleu"],
    ),

We have many predefined tasks in the catalog's :ref:`Tasks section <catalog.tasks>`.

If a cataloged task fits your use case, you may reference it by name:

.. code-block:: python

    task='tasks.translation.directed',

Loading the Dataset
---------------------

To load data from an external source, we use a loader.
For example, to load the `wmt16` translation dataset from the HuggingFace hub:

.. code-block:: python

    loader=LoadHF(path="wmt16", name="de-en"),

More loaders for different sources are available in the  :class:`loaders <unitxt.loaders>` section.

The Preprocessing pipeline
---------------------------

The preprocessing pipeline consists of operations to prepare your data according to the task's schema.

For example, to prepare the wmt16 dataset for translation task, we need to map the raw dataset field names to the standard
input and output fields of the task.  We also need to add new fields for the source and target language.

.. code-block:: python

    ...
    preprocess_steps=[
        CopyFields( # Copy the fields to prepare the fields required by the task schema
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

For custom operators refer to the :ref:`Operators Tutorial <adding_operator>`.

The Template
----------------

The responsibility of the template is to verbalize the task's input and output fields to input of the model and the gold references.
For example taking the input fields `text`, `source_language` and `target_language` and format as a prompt

`Translate this sentence from {source_language} to {target_language}: {text}.``

The template also verbalizes the output fields as gold references.  In Unitxt , references are the expected textual outputs of the model.
In this example, the `translation` field is taken, as is, as a gold reference.
However, in other cases , the output field may undergo some transformations.

If using a predefined task, you can choose from the corresponding templates available in the catalog's :ref:`Templates section <catalog.templates>`.

.. note::

   Use the :ref:`comprehensive guide on templates  <templates>` for more templates features.

Alternatively you can define your custom templates:

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

Once your card is ready, you can test it.  Here we use standard translation templates from
the Unitxt catalog.

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

The `test_card` function generates the dataset using all templates defined in the card within context learning mode and one demonstration.
It prints out three examples from the test fold, and runs the metrics defined on the datasets on
(1) randomly generated text
(2) text which is equal to one of the references.

Most metrics should return a low score (near 0) on random data and a score of 1 when the data is equal to the references.
Errors/warnings are printed if it's not the case.

Adding to the catalog
-----------------------

Once your card is ready and tested you can add it to the catalog.


.. code-block:: python

    from unitxt import add_to_catalog

    add_to_catalog(card, 'cards.wmt.en_de')

In the same way you can save also your custom templates and tasks.

.. note::
   By default, a new artifact is added to a local catalog stored
   in the library directory. To use a different catalog,
   use the `catalog_path` argument.

   In order to automatically load from your new catalog, remember to
   register your new catalog by `unitxt.register_catalog('my_catalog')`
   or by setting the `UNITXT_ARTIFACTORIES` environment variable to include your catalog.


Putting it all together!
------------------------

Now everything is ready to use the data! We use standard recipe to load it with three in-context examples.

.. code-block:: python

    from unitxt.standard import StandardRecipe
    from unitxt import load_dataset

    recipe = StandardRecipe(
        card='cards.wmt.en_de',
        num_demos=3, # The number of demonstrations for in-context learning
        demos_pool_size=100 # The size of the demonstration pool from which to sample the 5 demonstrations
        template_card_index=0 # Take the first template defined in the card
    )

    dataset = load_dataset(recipe)


Or even simpler with hugginface datasets:

.. code-block:: python

    from datasets import load_dataset

    dataset = load_dataset('unitxt/data', 'card=cards.wmt.en_de,num_demos=5,demos_pool_size=100,template_card_index=0')

And the same results as before will be obtained.

Sharing the Dataset
--------------------

Once the dataset is loaded, it may be shared with others by simply sharing the card file
with them to paste into their local catalog.

You may also submit a PR to integrate your new datasets into the official Unitxt release.
