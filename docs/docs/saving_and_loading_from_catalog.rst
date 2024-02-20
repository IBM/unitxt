.. _using_catalog:

=====================================
Saving and Loading From the Catalog
=====================================

Unitxt catalog is a place for pepole to share their processing tools with others.
Templates, formats, operators and more Unitxt assets can be shared thorugh a local catalog in a folder in the local filesystem or through a folder in a github repository.

Defining local catalog
----------------------

In order do define a local private catalog you can:

.. code-block:: python

    from unitxt import register_local_catalog

    register_local_catalog("path/to/catalog/directory")

Adding assets to a catalog
--------------------------

Once your catalog is registered you can save artifacts to the catalog:

.. code-block:: python

    from unitxt.task import FormTask
    from unitxt import save_to_catalog

    my_task = FormTask(...)

    catalog_name = "tasks.my_task"

    save_to_catalog(my_task, catalog_name, catalog_path="path/to/catalog/directory")

You can also save artifacts to the library default catalog:

.. code-block:: python

    save_to_catalog(my_task, catalog_name)


Using catalog assets
--------------------

In order to use catalog objects you just need to specify their name to the unitxt object that will use them.

For example now `tasks.my_task` can be used by the `StandardRecipe`:

.. code-block:: python

    from unitxt.card import TaskCard

    card = TaskCard(
        ...
        task="tasks.my_task"
    )

Modifying catalog assets on the fly
------------------------------------

If we want to get asset from the catalog but to edit his fields we can do it with a simple syntax:
`asset.name[key_to_modify=new_value]` we can also assign lists by `asset.name[key_to_modify=[new_value_0, new_value_1]]`
For example if we want to use a task but change its metric list we can:

.. code-block:: python

    from unitxt.card import TaskCard

    card = TaskCard(
        ...
        task="tasks.my_task[metrics=[matrics.accuracy, metrics.f1[reduction=median]]"
    )

Accessing catalog assets directly
------------------------------------

In order to access catalog assets directly we can use `get_from_catalog`

.. code-block:: python

    from unitxt import get_from_catalog

    my_task = get_from_catalog("tasks.my_task")


Using many catalogs
-------------------

Unitxt use by default many catalog such as the local library catalog and online community catalog hosted on github.
Assets are always taken from the last catalog registered that have the asset.

Defning catalog through environment variable
--------------------------------------------

In cases where unitxt is run by other application you might want to define your custom catalogs
thorugh an environment variable.

.. code-block:: bash

    export UNITXT_ARTIFACTORIES="path/to/first/catalog:path/to/second/catalog"


You can read more about catalogs here: :class:`catalog <unitxt.catalog>`.