.. _evaluating_datasets:

===================================
Evaluating Datasets
===================================

Evaluating a dataset 
-----------------

Evaluating a dataset can be done using the Huggingface metric API without direct installation of Unitxt:

.. code-block:: python

  import evaluate
  from transformers import pipeline
  from datasets import load_dataset

  dataset = load_dataset('unitxt/data', 'card=cards.wnli,template=templates.classification.multi_class.relation.default,max_test_instances=100',trust_remote_code=True)
  testset = dataset['test']
  model_inputs = testset['source']

  # These two lines can be replaces by any inference engine, that receives the model_input strings
  # and returns models predictions as string.
  model = pipeline(model='google/flan-t5-base')
  predictions = [output['generated_text'] for output in model(model_inputs,max_new_tokens=30)]
  
  metric = evaluate.load('unitxt/metric')
  dataset_with_scores = metric.compute(predictions=predictions,references=testset)

The following prints the scores defined in WNLI task (f1_micro, f1_macro, accuracy, as well as their confidence intervals).

.. code-block:: python
  
    [print(item) for item in dataset_with_scores[0]['score']['global'].items()] 


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