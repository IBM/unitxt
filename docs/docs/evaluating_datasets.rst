.. _evaluating_datasets:

===================================
Evaluating Datasets
===================================

Unitxt can be used to evaluate datasets from it's catalog and user defined datasets.

.. code-block:: python

  from unitxt import load_dataset, evaluate
  from unitxt.inference import HFAutoModelInferenceEngine

  dataset = load_dataset(
    card="cards.wnli",
    format="formats.chat_api",
    max_test_instances=100,
    split="test"
  )

  # The following lines can be replaced by any inference engine that receives the model_input
  # (found in dataset['source']) and returns model predictions as string.
  model = HFAutoModelInferenceEngine(
      model_name="Qwen/Qwen1.5-0.5B-Chat", max_new_tokens=32
  )
  predictions = model(dataset)

  results = evaluate(predictions, dataset)

The following prints the scores defined in WNLI task (f1_micro, f1_macro, accuracy, as well as their confidence intervals).

.. code-block:: python

    print(results.global_scores.summary)


.. code-block::

  | score_name        |   score | ci_low   | ci_high   |
  |:------------------|--------:|:---------|:----------|
  | accuracy          |    0.56 | 0.44     | 0.68      |
  | f1_entailment     |    0.73 |          |           |
  | f1_macro          |    0.36 | 0.32     | 0.41      |
  | f1_micro          |    0.57 | 0.48     | 0.7       |
  | f1_not entailment |    0    |          |           |
  | score             |    0.57 | 0.48     | 0.7       |
  Main Score: f1_micro
  Num Instances: 71


If you want to evaluate with few templates or few num_demos you can run:

.. code-block:: python

  dataset = load_dataset(
    card="cards.wnli",
    template=["templates.classification.multi_class.relation.default","templates.key_val"],
    num_demos=[0,1,3],
    demos_pool_size=10,
    max_test_instances=100
  )

This will randomly sample from the templates and for each instance assign a random template from the list and run number of demonstration from the list.

If you the want to explore the score per template and num of demonstrations you can add ``group_by=["template", "num_demos"]``.
If you want to get the score for each combination you should add ``group_by=[["template", "num_demos"]]`` or if you want for each group and for each combination you caption
add them all together ``group_by=["template", "num_demos", ["template", "num_demos"]]`` or in a full recipe:

.. code-block:: python

  dataset = load_dataset(
    card="cards.wnli",
    template=["templates.classification.multi_class.relation.default","templates.key_val"],
    num_demos=[0,1,3],
    group_by=["template","num_demos",["template","num_demos"]],
    demos_pool_size=10,
    max_test_instances=100
  )

The grouping can be done based on any field of the task or the metadata, so for classification task you can also group by label with ``group_by=["label"]``.

.. code-block:: python

    print(results.groups_scores.summary)

Will print:
.. code-block::

    # Group By: template
    | template                                              |    score | score_name   |   score_ci_low |   score_ci_high |   num_of_instances |
    |:------------------------------------------------------|---------:|:-------------|---------------:|----------------:|-------------------:|
    | templates.classification.multi_class.relation.default | 0.264151 | f1_micro     |       0.137052 |        0.421053 |                 41 |
    | templates.key_val                                     | 0.210526 | f1_micro     |       0.06367  |        0.388275 |                 30 |

    # Group By: num_demos
    |   num_demos |    score | score_name   |   score_ci_low |   score_ci_high |   num_of_instances |
    |------------:|---------:|:-------------|---------------:|----------------:|-------------------:|
    |           1 | 0.30303  | f1_micro     |      0.125     |        0.486229 |                 23 |
    |           3 | 0.275862 | f1_micro     |      0.0769231 |        0.478979 |                 22 |
    |           0 | 0.137931 | f1_micro     |      0         |        0.343992 |                 26 |

    # Group By: template, num_demos
    | template                                              |   num_demos |    score | score_name   | score_ci_low        | score_ci_high      |   num_of_instances |
    |:------------------------------------------------------|------------:|---------:|:-------------|:--------------------|:-------------------|-------------------:|
    | templates.classification.multi_class.relation.default |           1 | 0.333333 | f1_micro     | 0.08606627464804656 | 0.5990125628603442 |                 16 |
    | templates.key_val                                     |           3 | 0.272727 | f1_micro     | 0.09226935524612535 | 0.5454545454545454 |                 16 |
    | templates.key_val                                     |           1 | 0.222222 | f1_micro     | 0.0                 | 0.7225818346056374 |                  7 |
    | templates.classification.multi_class.relation.default |           3 | 0.285714 | f1_micro     | 0.0                 | 0.779447856172277  |                  6 |
    | templates.classification.multi_class.relation.default |           0 | 0.181818 | f1_micro     | 0.0                 | 0.4105379478071894 |                 19 |
    | templates.key_val                                     |           0 | 0        | f1_micro     |                     |                    |                  7 |


Metadata
--------
The result object that returned by `evaluate` function contains `metadata` feature.
This feature contains the dataset and the inference engine metadata (if exists).:

This metadata can be accessed and used for further analysis or debugging.