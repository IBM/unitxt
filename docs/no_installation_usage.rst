===================================
Loading Datasets
===================================

Loading a dataset
-----------------

With unitxt you can load a dataset without installing the package simply
by using the following code:

.. code-block:: python

  from datasets import load_dataset

  dataset = load_dataset('unitxt/data', 'card=cards.wnli')

.. code-block:: python

  print(dataset)
  DatasetDict({
    train: Dataset({
        features: ['metrics', 'source', 'target', 'references', 'group', 'postprocessors'],
        num_rows: 599
    })
    validation: Dataset({
        features: ['metrics', 'source', 'target', 'references', 'group', 'postprocessors'],
        num_rows: 36
    })
    test: Dataset({
        features: ['metrics', 'source', 'target', 'references', 'group', 'postprocessors'],
        num_rows: 71
    })
  })

This prints the source text (input to the model) of the first sample in the training set:

.. code-block:: python

    print(dataset['train'][0]['source'])

    'Given this sentence: I stuck a pin through a carrot. When I pulled the pin out, it had a hole., classify if this sentence: The carrot had a hole. is entailment, not entailment.\n'

This prints the reference text (expected output of the model) of the first sample in the training set:

.. code-block:: python

    print(dataset['train'][0]['references'][0])
    'not entailment'


Loading a customized datasets
-----------------------------

Unitxt enable the loading of datasets in different forms if they have a dataset card stored in
the unitxt catalog. For example, here we load wnli in 3 shots format:

.. code-block:: python

  dataset = load_dataset('unitxt/data', 'card=cards.wnli,num_demos=3,demos_pool_size=100')

Now the source source text (input to the model) of the first sample in the training set has in-context examples:

.. code-block:: python

    print(dataset['train'][0]['source'])

    """Given this sentence: Alice was dusting the living room and trying to find the button that Mama had hidden. No time today to look at old pictures in her favorite photo album. Today she had to hunt for a button, so she put the album on a chair without even opening it., classify if this sentence: She put the album on a chair without even opening the living room. is entailment, not entailment.
    entailment

    Given this sentence: The trophy doesn't fit into the brown suitcase because it is too large., classify if this sentence: The suitcase is too large. is entailment, not entailment.
    entailment

    Given this sentence: The foxes are getting in at night and attacking the chickens. They have gotten very bold., classify if this sentence: The foxes have gotten very bold. is entailment, not entailment.
    not entailment

    Given this sentence: Grace was happy to trade me her sweater for my jacket. She thinks it looks dowdy on her., classify if this sentence: The sweater looks dowdy on her. is entailment, not entailment."""
