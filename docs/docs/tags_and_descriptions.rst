.. _tags_and_descriptions:

.. note::

   To use this tutorial, you need to :ref:`install unitxt <install_unitxt>`.

=====================================
Tags and Descriptions
=====================================

Artifacts in the Unitxt catalog, such as datasets, templates, formats, and operators, can be valuable to others. To help others discover and understand them, Unitxt provides an option to add tags and descriptions to catalog assets.

To search for a catalog asset by tag or description, use the text-based search located in the top-right corner of the Unitxt website.

Adding Descriptions and Tags for Catalog Assets
-----------------------------------------------

Each Unitxt asset has two dedicated fields for tags and descriptions: `__tags__` and `__description__`.

You can assign their values while constructing a Unitxt asset and then save them to the catalog.

For example, if you want to add tags and a description to the `wikitq` dataset card:

.. code-block:: python

    card = TaskCard(
        loader=LoadHF(path="wikitablequestions"),
        task="tasks.qa.with_context.extractive",
        templates="templates.qa.with_context.all",
        __description__="The WikiTableQuestions dataset is a large-scale dataset for the task of question answering on semi-structured tables.",
        __tags__={
            "modality": "table",
            "urls": {"arxiv": "https://arxiv.org/abs/1508.00305"},
            "languages": ["english"],
        },
    )

You can then save the card to the catalog with:

.. code-block:: python

    add_to_catalog(card, "cards.wikitq", overwrite=True)

As a result, the description and tags will appear on the catalog webpage for `cards.wikitq`, as seen here: :ref:`WikiTQ <catalog.cards.wikitq>`.

Editing Existing Assets to Add Tags or Descriptions
---------------------------------------------------

High-quality tags and descriptions enrich the Unitxt catalog.

You can find the preparation code for any Unitxt asset at: https://github.com/IBM/unitxt/tree/main/prepare

Choose a Unitxt asset, add information about it, and then submit a Pull Request with your changes.

How to Write Good Descriptions and Tags
---------------------------------------

1. **Description**: Keep it brief and include all essential information needed to understand the asset's purpose.
2. **Tags**: Classify the asset based on its main aspects, such as `domain`, `task`, `language`, `modality`, tested `skill`, etc.
