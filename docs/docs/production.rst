.. _production:

.. note::

   To use this tutorial, you need to :ref:`install unitxt <install_unitxt>`.

=====================================
Using in production
=====================================

Unitxt can be used to process data in production. First define a recipe:

.. code-block:: python

  recipe = "card=cards.wnli,template=templates.classification.multi_class.relation.default,demos_pool_size=5,num_demos=2"


Second prepare an instance in the exact schema of the task in that recipe:


.. code-block:: python

  instance = {
    "label": "?",
    "text_a": "It works perfectly",
    "text_b": "It works!",
    "classes": ["entailment", "not entailment"],
    "type_of_relation": "entailment",
    "text_a_type": "premise",
    "text_b_type": "hypothesis",
  }

Then you can produce that model-ready data with the `produce` function:

.. code-block:: python

  from unitxt import produce

  result = produce(instance, recipe)

Then you have the production ready instance in the result. If you `print(instance["source"])` you will get:

.. code-block::

    Given a premise and hypothesis classify the entailment of the hypothesis to one of entailment, not entailment.

    premise: When Tatyana reached the cabin, her mother was sleeping. She was careful not to disturb her, undressing and climbing back into her berth., hypothesis: mother was careful not to disturb her, undressing and climbing back into her berth.
    The entailment class is entailment

    premise: The police arrested all of the gang members. They were trying to stop the drug trade in the neighborhood., hypothesis: The police were trying to stop the drug trade in the neighborhood.
    The entailment class is not entailment

    premise: It works perfectly, hypothesis: It works!
    The entailment class is




