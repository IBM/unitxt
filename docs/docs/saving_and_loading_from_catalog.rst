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

To modify a catalog asset's fields dynamically, upon fetching the asset from the catalog, and instantiating it into a python object, use the syntax: ``asset.name[key_to_modify=new_value]``. 
To assign lists, use: ``asset.name[key_to_modify=[new_value_0, new_value_1]]``. 
To assign dictionaries, use: ``asset.name[key_to_modify={new_key_0=new_value_0,new_key_1=new_value_1}]``.
Note that the whole new value of the field has to be specified; not just one item of a list, or one key of the dictionary.
For instance, to change the metric list and the reference specification of a task:

.. code-block:: python

    from unitxt.card import TaskCard

    card = TaskCard(
        ...
        task="tasks.my_task[metrics=[metrics.accuracy, metrics.f1[reduction=median]],reference_fields={output=int}]"
    )

Accessing Catalog Assets Directly
---------------------------------

Use ``get_from_catalog`` to directly access catalog assets, and obtain an asset instantiated as a python object:

.. code-block:: python

    from unitxt import get_from_catalog

    my_task = get_from_catalog("tasks.my_task")

A Catalog Asset Linking to Another Catalog Asset
------------------------------------------------

A catalog asset can be just a link to another asset. This feature comes handy when for some reason, ``asset1`` -- the name of an asset, which reflects its place in the catalog, is changed to ``asset2``, while much code already exists where the old name of the asset, ``asset1`` is hard coded.
In such a case, an asset of type :class:`ArtifactLink <unitxt.artifact.ArtifactLink>`, that links to ``asset2``, can take the place of ``asset1`` in the catalog. 
When ``asset1`` is accessed from an existing code, Unixt Catalog realizes that the asset fetched from position ``asset1`` is an ``ArtifactLink``, so it continues to ``asset2`` -- the Artifact linked to by ``asset1``, instantiates and returns it.
If that linked-to asset, ``asset2``, turns out to be an ``ArtifactLink`` as well, Unitxt Catalog continues along the links, until a non-link Artifact is reached, and that one is instantiated as a python object and returned.

Using Multiple Catalogs
-----------------------

By default, Unitxt uses several catalogs, such as the local library catalog and online community catalogs hosted on GitHub. Assets are sourced from the last registered catalog containing the asset.

Defining Catalogs Through Environment Variables
-----------------------------------------------

When Unitxt is executed by another application, you might need to specify custom catalogs through an environment variable:

.. code-block:: bash

    export UNITXT_CATALOGS="path/to/first/catalog:path/to/second/catalog"

Learn more about catalogs here: :class:`catalog <unitxt.catalog>`.
