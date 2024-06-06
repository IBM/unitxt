.. _debugging:

===================================
Debugging Unitxt
===================================

Debugging cards
----------------

To help test and debug cards, there is a utility function called test_card imported from `unitxt.test_utils.card`. 

.. code-block:: python

  from unitxt.test_utils.card import test_card

  card = TaskCard(
    ...
  )
  test_card(card)


By default, the function generates the dataset using all templates defined in the card.
For each template, it prints out up to 5 examples from the test fold.  For each example,
you can see all the fields in the dataset.

In the rest of this tutorial we will review the output of the `test_card` function
of the `universal_ner` card.

.. code-block:: bash
  
  prepare/cards/universal_ner.py

::

  Loading limited to 30 instances by setting LoadHF.loader_limit;
  ----------
  Showing up to 5 examples from stream 'test':
  
  metrics (list):
      ['metrics.ner']
  data_classification_policy (list):
      ['public']
  source (str):
      From the following text, identify spans with entity type:Person, Organization, Location.
      text: What is this Miramar?
  target (str):
      Miramar: Location
  references (list):
      ['Miramar: Location']
  task_data (str):
      {"text": "What is this Miramar?", "text_type": "text", "class_type": "entity type", "classes": ["Person", "Organization", "Location"], "spans_starts": [13], "spans_ends": [20], "labels": ["Location"], "metadata": {"template": "templates.span_labeling.extraction.identify"}}
  group (str):
      unitxt
  postprocessors (list):
      ['processors.to_span_label_pairs']



The code then runs the metrics defined on the datasets on 

1.  predictions which is equal to one of the references. 

2.  random text predictions

To help validate the post processing of the predictions and references , the code prints the post processed values.
For example, we can see how the string "Miramar: Location" is parsed by the post processors to a list of tuples.



::

  ****************************************
  Running with the gold references as predictions.
  Showing the output of the post processing:
  *****
  Prediction: (str)     Miramar: Location
  Processed prediction: (list) [('Miramar', 'Location')]
  Processed references: (list) [[('Miramar', 'Location')]]
  *****
  Prediction: (str)     Argentina: Location
  Processed prediction: (list) [('Argentina', 'Location')]
  Processed references: (list) [[('Argentina', 'Location')]]
  *****
  Prediction: (str)     None
  Processed prediction: (list) []
  Processed references: (list) [[]]
  *****
  Prediction: (str)     Argentina: Location
  Processed prediction: (list) [('Argentina', 'Location')]
  Processed references: (list) [[('Argentina', 'Location')]]
  *****
  Prediction: (str)     None
  Processed prediction: (list) []
  Processed references: (list) [[]]
  *****
  
  *****
  Score output:
  {
      "global": {
          "f1_Location": 1.0,
          "f1_macro": 1.0,
          "f1_micro": 1.0,
          "f1_micro_ci_high": NaN,
          "f1_micro_ci_low": NaN,
          "in_classes_support": 1.0,
          "precision_macro": 1.0,
          "precision_micro": 1.0,
          "recall_macro": 1.0,
          "recall_micro": 1.0,
          "score": 1.0,
          "score_ci_high": NaN,
          "score_ci_low": NaN,
          "score_name": "f1_micro"
      },
      "instance": {
          "f1_Location": 1.0,
          "f1_macro": 1.0,
          "f1_micro": 1.0,
          "in_classes_support": 1.0,
          "precision_macro": 1.0,
          "precision_micro": 1.0,
          "recall_macro": 1.0,
          "recall_micro": 1.0,
          "score": 1.0,
          "score_name": "f1_micro"
      }
  }


Most metrics should return a low score (near 0) on random text and a score of 1 when the data is equal to the references.
Errors/warnings are printed if it's not the case. 

If you want to disable the these tests, set ``test_exact_match_score_when_predictions_equal_references=False`` and/or 
``test_full_mismatch_score_with_full_mismatch_prediction_values=False``.   

You can set the expected scores using the following parameters:

1. ``exact_match_score``: The expected score to be returned when predictions are equal the gold reference. Default is 1.0.

2. ``maximum_full_mismatch_score``: The maximum score allowed to be returned when predictions are full mismatched. Default is 0.0.

3. ``full_mismatch_prediction_values``: An optional list of prediction values to use for testing full mismatches. If not set, a default set of values: ["a1s", "bfsdf", "dgdfgs", "gfjgfh", "ghfjgh"] is used.

If you want to generate the card with different parameters, they can be provided as additional
arguments to the test_card() function.

.. code-block:: python

  # Test the templates with few shots
  test_card(card,num_demos=1,demo_pool_size=10)

test_card has an optional parameter flag debug. When set to True, the card is executed in debug mode, one step at a time. For example, it starts with loading the dataset, then performing the defined preprocessing steps, then performing the template rendering steps. 
After each step it prints the number of instances in each split, and one example from each split.

.. code-block:: python
  # Shows the step by step processing of data.
  test_card(card,debug=True)

If you get an error, it's best that you turn this flag on, and see where in the execution flow it happens. It's also a good way if want to understand exactly how datasets are generated and what each step performs.

Increase log verbosity
----------------------

If you want to get more information during the run (for example, which artifict are loaded from which catalog),
you can set the UNITXT_DEFAULT_VERBOSITY environment variable or modify the global setting in the code.

.. code-block:: bash

  env UNITXT_DEFAULT_VERBOSITY=debug python prepare/cards/wnli.py

.. code-block:: python

  from .settings_utils import get_settings
  settings = get_settings()
  settings.default_verbosity = "debug"
