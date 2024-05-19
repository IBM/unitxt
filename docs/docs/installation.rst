.. _install_unitxt:
==============
Installation
==============

Unitxt conforms to the Huggingface datasets and metrics API, so it can be used without explicitly installing the unitxt package.

.. code-block:: python

  import evaluate
  from datasets import load_dataset
  from transformers import pipeline


  dataset = load_dataset('unitxt/data', 'card=cards.wnli,template=templates.classification.multi_class.relation.default,max_test_instances=20',trust_remote_code=True)
  testset = dataset["test"]
  model_inputs = testset["source"]
  model = pipeline(model='google/flan-t5-base')
  predictions = [output['generated_text'] for output in model(model_inputs,max_new_tokens=30)]
  
  metric = evaluate.load("unitxt/metric",trust_remote_code=True)
  scores = metric.compute(predictions=predictions,references=testset)

Note, the `trust_remote_code=True` flag is required because in the background the Huggingface API downloads and installs the
latest version of unitxt.  The core of unitxt has minimal dependencies (none beyond Huggingface evaluate).
Note that specific metrics or other operators, may required specific dependencies, which are checked before the first time they are used.
An error message is printed if the there are missing installed dependencies.

The benefit of using the Huggingface API approach is that you can load a Unitxt dataset, just like every other Huggingface dataset, 
so it can be used in preexisting code without modifications.  
However, this incurs extra overhead when Huggingface downloads the unitxt package and does not expose all unitxt capabilities
(e.g. defining new datasets, metrics, templates, and more)

To get the full capabilities of Unitxt , install Unitxt locally from pip:

.. code-block:: bash

  pip install unitxt


You can then use the API:

.. code-block:: python

  from unitxt import load_dataset,evaluate
  from unitxt.inference import HFPipelineBasedInferenceEngine

  dataset = load_dataset('card=cards.wnli,template=templates.classification.multi_class.relation.default,max_test_instances=20')
  test_dataset = dataset["test"]

  model_name="google/flan-t5-large"
  inference_model = HFPipelineBasedInferenceEngine(model_name=model_name, max_new_tokens=32)
  predictions = inference_model.infer(test_dataset)

  scores = evaluate(predictions=predictions, data=test_dataset)

