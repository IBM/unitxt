.. _loading_datasets:

===================================
Loading Datasets
===================================


.. code-block:: python

  from unitxt import load_dataset

  dataset = load_dataset(card="cards.wnli")

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

The following prints the source text (input to the model) of the first sample in the training set:

.. code-block:: python

    print(dataset['train'][0]['source'])

.. code-block::

    Given a premise and hypothesis classify the entailment of the hypothesis to one of entailment, not entailment.
    premise: Grace was happy to trade me her sweater for my jacket. She thinks it looks dowdy on her.
    hypothesis: The sweater looks dowdy on her.
    The entailment class is

This prints the reference text (expected output of the model) of the first sample in the training set:

.. code-block:: python

    print(dataset['train'][0]['references'][0])

.. code-block::

    'not entailment'


Loading a customized dataset
----------------------------

Unitxt enables formatting the dataset in different ways.

As example, here we load wnli in 3 shots format:

.. code-block:: python

  from unitxt import load_dataset
  dataset = load_dataset(card="cards.wnli",template="templates.classification.multi_class.relation.default",num_demos=3,demos_pool_size=100)

Now the source text (input to the model) of the first sample in the training set has in-context examples:

.. code-block:: python

    print(dataset['train'][0]['source'])

.. code-block::

    Given a premise and hypothesis classify the entailment of the hypothesis to one of entailment, not entailment.
    premise: The journalists interviewed the stars of the new movie. They were very cooperative, so the interview lasted for a long time.
    hypothesis: The journalists were very cooperative, so the interview lasted for a long time.
    The entailment class is entailment

    premise: The table won't fit through the doorway because it is too narrow.
    hypothesis: The table is too narrow.
    The entailment class is entailment

    premise: Sam pulled up a chair to the piano, but it was broken, so he had to stand instead.
    hypothesis: The chair was broken, so he had to stand instead.
    The entailment class is not entailment

    premise: Grace was happy to trade me her sweater for my jacket. She thinks it looks dowdy on her.
    hypothesis: The sweater looks dowdy on her.
    The entailment class is

Loading a Dataset with Multiple Templates or Number of Demonstrations
---------------------------------------------------------------------

You can sample a template for each instance from a pool of templates by assigning the ``template`` argument a list of templates. Similarly, you can sample the number of demonstrations by assigning ``num_demos`` a list of integers.

Here is an example of using random templates and a varying number of demonstrations for each instance of the data:

.. code-block:: python

  dataset = load_dataset(card="cards.wnli",template=["templates.classification.multi_class.relation.default","templates.key_val"],num_demos=[0,1,3],demos_pool_size=100)
