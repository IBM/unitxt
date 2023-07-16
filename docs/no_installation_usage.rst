===================================
Loading Premade Dataset Recipes
===================================

Loading a Premade Dataset Recipe
---------------------------------

With unitxt you can load a dataset without installing the package simply 
by using the following code (loading wnli dataset in 3 shot format):

.. code-block:: python

  from datasets import load_dataset

  dataset = load_dataset('unitxt/data', 'recipes.wnli_3_shot')

Loading Customized Data
--------------------------

Unitxt enable the loading of datasets in different forms if they have a dataset card stored in 
the unitxt catalog. For example, here we load wnli in 5 shots format:

.. code-block:: python

  dataset = load_dataset('unitxt/data', 'card=cards.wnli,num_demos=5,demos_pool_size=100')