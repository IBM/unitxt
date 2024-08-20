.. _using_catalog:

=====================================
Save/Load from Catalog
=====================================

The Unitxt catalog serves as a repository for people to share their processing tools. This includes templates, formats, operators, and other Unitxt assets. These can be shared through a local catalog located in a directory on the local filesystem or via a directory in a GitHub repository.

Defining a Local Catalog
------------------------

To define a local, private catalog, use the following code:

.. code-block:: python

    from unitxt import register_local_catalog

    register_local_catalog("path/to/catalog/directory")

Adding Assets to the Catalog
----------------------------

Once your catalog is registered, you can add artifacts to it:

.. code-block:: python

    from unitxt.task import Task
    from unitxt import add_to_catalog

    my_task = Task(...)

    catalog_name = "tasks.my_task"

    add_to_catalog(my_task, catalog_name, catalog_path="path/to/catalog/directory")

It's also possible to add artifacts to the library's default catalog:

.. code-block:: python

    add_to_catalog(my_task, catalog_name)

Using Catalog Assets
--------------------

To use catalog objects, simply specify their name in the Unitxt object that will use them. 

.. code-block:: python

    from unitxt.card import TaskCard

    card = TaskCard(
        ...
        task="tasks.my_task"
    )

Modifying Catalog Assets on the Fly
-----------------------------------

To modify a catalog asset's fields dynamically, use the syntax: `asset.name[key_to_modify=new_value]`. To assign lists, use: `asset.name[key_to_modify=[new_value_0, new_value_1]]`. For instance, to change the metric list of a task:

.. code-block:: python

    from unitxt.card import TaskCard

    card = TaskCard(
        ...
        task="tasks.my_task[metrics=[metrics.accuracy, metrics.f1[reduction=median]]]"
    )

Accessing Catalog Assets Directly
---------------------------------

Use `get_from_catalog` to directly access catalog assets:

.. code-block:: python

    from unitxt import get_from_catalog

    my_task = get_from_catalog("tasks.my_task")

Using Multiple Catalogs
-----------------------

By default, Unitxt uses several catalogs, such as the local library catalog and online community catalogs hosted on GitHub. Assets are sourced from the last registered catalog containing the asset.

Defining Catalogs Through Environment Variables
-----------------------------------------------

When Unitxt is executed by another application, you might need to specify custom catalogs through an environment variable:

.. code-block:: bash

    export UNITXT_ARTIFACTORIES="path/to/first/catalog:path/to/second/catalog"

Learn more about catalogs here: :class:`catalog <unitxt.catalog>`.
