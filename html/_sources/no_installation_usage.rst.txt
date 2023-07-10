=======
Usage
=======

Loading a Dataset (In the future)
----------------------------------

With unitxt you can load a dataset without installing the package simply 
by using the following code:

.. code-block:: python

  from datasets import load_dataset

  dataset = load_dataset('ibm/unitxt', 'squad')

Loading a Dataset with Different Format or Instructions
-------------------------------------------------------

.. note:: This section is to be determined (TBD).

Loading a dataset with 5 demonstrations for in context learning
--------------------------------------------------------------------------------

.. code-block:: python

  dataset = load_dataset('ibm/unitxt', 'card=squad,num_demos=5')