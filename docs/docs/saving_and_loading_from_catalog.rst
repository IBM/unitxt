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
To modify a catalog asset's fields dynamically, upon fetching the asset from the catalog, use the syntax: ``artifact_name[key_to_modify=new_value]``. 
To assign lists, use: ``asset.name[key_to_modify=[new_value_0, new_value_1]]``. 
To assign dictionaries, use: ``asset.name[key_to_modify={new_key_0=new_value_0,new_key_1=new_value_1}]``.
Note that the whole new value of the field has to be specified; not just one item of a list, or one key of the dictionary.
For instance, to change the metric specification of a task:

.. code-block:: python

    from unitxt.card import TaskCard

    card = TaskCard(
        ...
        task="tasks.my_task[metrics=[metrics.accuracy, metrics.f1[reduction=median]]]"
    )

Accessing Catalog Assets Directly
---------------------------------

Use ``get_from_catalog`` to directly access catalog assets, and obtain an asset instantiated as a python object of type ``unitxt.Artifact``:

.. code-block:: python

    from unitxt import get_from_catalog

    my_task = get_from_catalog("tasks.my_task")

A Catalog Asset Linking to Another Catalog Asset
------------------------------------------------

A catalog asset can be just a link to another asset. 
This feature comes handy when for some reason, we want to change the catalog name 
of an existing asset (e.g. ``asset1`` to ``asset2``), while there is already code 
that uses the old name of the asset and we want to avoid non-backward compatible changes.

In such a case, we can save the asset as ``asset2``, create an asset of type 
:class:`ArtifactLink <unitxt.artifact.ArtifactLink>` that links to ``asset2``, and save
that one as ``asset1``.
When ``asset1`` is accessed from an existing code, Unixt Catalog realizes that the asset fetched from position ``asset1`` 
is an ``ArtifactLink``, so it continues and fetches ``asset2`` -- the Artifact linked to by ``asset1``. 

.. code-block:: python

    link_to_asset2 = ArtifactLink(artifact_linked_to="asset2")
    add_to_catalog(
        link_to_asset2,
        "asset1",
        overwrite=True,
    )

Deprecated Asset
~~~~~~~~~~~~~~~~
Every asset has a special field named ``__deprecated_msg__`` of type ``str``, whose default value is None.
When None, the asset is cocnsidered non-deprecated. When not None, the asset is considered deprecated, and 
its ``__deprecated_msg__`` is logged at level WARN upon its instantiation. (Other than this logging, 
the artifact is instantiated normally.)

Combining this feature with ``ArtifactLink`` in the above example, we can also log a warning to the accessing code that 
the name ``asset1`` is to be replaced by ``asset2``. 

.. code-block:: python

    link_to_asset2 = ArtifactLink(artifact_linked_to="asset2",
           __deprecated_msg__="'asset1' is going to be deprecated. In future uses, please access 'asset2' instead.")
    add_to_catalog(
        link_to_asset2,
        "asset1",
        overwrite=True,
    )


Using Multiple Catalogs
-----------------------

By default, Unitxt uses several catalogs, such as the local library catalog and online community catalogs hosted on GitHub. Assets are sourced from the last registered catalog containing the asset.

Defining Catalogs Through Environment Variables
-----------------------------------------------

When Unitxt is executed by another application, you might need to specify custom catalogs through an environment variable:

.. code-block:: bash

    export UNITXT_CATALOGS="path/to/first/catalog:path/to/second/catalog"

Learn more about catalogs here: :class:`catalog <unitxt.catalog>`.
