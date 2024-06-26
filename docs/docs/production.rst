.. _production:

.. note::

   To use this tutorial, you need to :ref:`install unitxt <install_unitxt>`.

========================
Inference and Production
========================

In this guide you will learn how to use unitxt data recipes in production.

For instance, you learn how to make end-to-end functions like `paraphrase()`:

.. code-block:: python

  def paraphrase(text):
    return unitxt.infer(
      [{"input_text": text, "output_text": ""}],
      recipe="card=cards.coedit.paraphrase,template=templates.rewriting.paraphrase.default",
      engine="engines.model.flan.t5_small.hf"
    )

Which then can be used like:

.. code-block:: python

  paraphrase("So simple to paraphrase!")

In general, Unitxt is capable of:
 - Producing processed data according to a given recipe.
 - Post-processing predictions based on a recipe.
 - Performing end-to-end inference using a recipe and a specified inference engine.

Produce Data
------------

First, define a recipe:

.. code-block:: python

  recipe = "card=cards.wnli,template=templates.classification.multi_class.relation.default,demos_pool_size=5,num_demos=2"

Next, prepare a Python dictionary that matches the schema required by the recipe:

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

Then, produce the model-ready input data with the `produce` function:

.. code-block:: python

  from unitxt import produce

  result = produce(instance, recipe)

To view the formatted instance, print the result:

.. code-block::

  print(result["source"])

This will output instances like:

.. code-block::

    Given a premise and a hypothesis, classify the entailment of the hypothesis as either 'entailment' or 'not entailment'.

    premise: When Tatyana reached the cabin, her mother was sleeping. She was careful not to disturb her, undressing and climbing back into her berth., hypothesis: mother was careful not to disturb her, undressing and climbing back into her berth.
    The entailment class is entailment

    premise: The police arrested all of the gang members. They were trying to stop the drug trade in the neighborhood., hypothesis: The police were trying to stop the drug trade in the neighborhood.
    The entailment class is not entailment

    premise: It works perfectly, hypothesis: It works!
    The entailment class is

Post Process Data
-----------------

After obtaining predictions, they can be post-processed:

.. code-block:: python

  from unitxt import post_process

  prediction = model.generate(result["source"])
  processed_result = post_process(predictions=[prediction], data=[result])[0]

End to End Inference Pipeline
-----------------------------

You can also implement an end-to-end inference pipeline using your preferred data and an inference engine:

.. code-block:: python

  from unitxt import infer
  from unitxt.inference import HFPipelineBasedInferenceEngine

  engine = HFPipelineBasedInferenceEngine(
      model_name="google/flan-t5-small", max_new_tokens=32
  )

  infer(instance, recipe, engine)

Alternatively, you can specify any inference engine from the catalog:

.. code-block:: python

  infer(
    instance,
    recipe="card=cards.wnli,template=templates.classification.multi_class.relation.default,demos_pool_size=5,num_demos=2",
    engine="engines.model.flan.t5_small.hf"
  )
