.. _evaluating_datasets:

===================================
Evaluating Datasets
===================================

Unitxt can be used to evaluate datasets from it's catalog and user defined datasets.

.. code-block:: python

  from unitxt import load_dataset, evaluate

  dataset = load_dataset(card="cards.wnli",format="formats.chat_api",max_test_instances=100,split="test")

  # The following lines can be replaced by any inference engine that receives the model_input
  # (found in dataset['source']) and returns model predictions as string.

  from unitxt.inference import HFPipelineBasedInferenceEngine
  model = HFPipelineBasedInferenceEngine(
      model_name="Qwen/Qwen1.5-0.5B-Chat", max_new_tokens=32
  )
  predictions = model(dataset)

  results = evaluate(predictions, dataset)

The following prints the scores defined in WNLI task (f1_micro, f1_macro, accuracy, as well as their confidence intervals).

.. code-block:: python

    print(results.global_scores.summary)


.. code-block::

    ('f1_macro', 0.393939393939394)
    ('f1_entailment', 0.787878787878788)
    ('f1_not entailment', 0.0)
    ('score', 0.65)
    ('score_name', 'f1_micro')
    ('score_ci_low', 0.4000000000000001)
    ('score_ci_high', 0.8000000000000002)
    ('f1_macro_ci_low', 0.28571428571428575)
    ('f1_macro_ci_high', 0.4444444444444445)
    ('accuracy', 0.65)
    ('accuracy_ci_low', 0.4)
    ('accuracy_ci_high', 0.85)
    ('f1_micro', 0.65)
    ('f1_micro_ci_low', 0.4000000000000001)
    ('f1_micro_ci_high', 0.8000000000000002)


If you want to evaluate with few templates or few num_demos you can run:

.. code-block:: python

  dataset = load_dataset(card="cards.wnli",template=["templates.classification.multi_class.relation.default","templates.key_val"],num_demos=[0,1,3],demos_pool_size=10,max_test_instances=100)

This will randomly sample from the templates and for each instance assign a random template from the list and run number of demonstration from the list.

If you the want to explore the score per template and num of demonstrations you can add ``group_by=["template", "num_demos"]``.
If you want to get the score for each combination you should add ``group_by=[["template", "num_demos"]]`` or if you want for each group and for each combination you caption
add them all together ``group_by=["template", "num_demos", ["template", "num_demos"]]`` or in a full recipe:

.. code-block:: python

  dataset = load_dataset(card="cards.wnli",template=["templates.classification.multi_class.relation.default","templates.key_val"],num_demos=[0,1,3],group_by=["template","num_demos",["template","num_demos"]],demos_pool_size=10,max_test_instances=100)

The grouping can be done based on any field of the task or the metadata, so for classification task you can also group by label with ``group_by=["label"]``.
