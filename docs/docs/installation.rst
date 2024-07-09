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
  dataset_with_scores = metric.compute(predictions=predictions,references=testset)
  [print(item) for item in scores[0]['score']['global'].items()] 

Note, the `trust_remote_code=True` flag is required because in the background the Huggingface API downloads and installs the
latest version of Unitxt from https://huggingface.co/datasets/unitxt/data/tree/main.

The core of Unitxt has minimal dependencies (none beyond Huggingface evaluate).
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

  dataset_with_scores = evaluate(predictions=predictions, data=test_dataset)
  [print(item) for item in dataset_with_scores[0]['score']['global'].items()] 


.. warning::
   It's important not to mix calls to the Unitxt directs APIs and the Huggingface APIs in the same program.  Use either
   the direct Unitxt APIs or the Huggingface APIs to load datasets and metrics.

If you get an error message like:

.. code-block::

   datasets_modules.datasets.unitxt--data.df049865776d8814049d4543a4068e50cda79b1558dc933047f4a41d087cc120.hf_utils.UnitxtVersionsConflictError:
   Located installed unitxt version 1.9.0 that is older than unitxt Huggingface dataset version 1.10.0.

It means that you are loading datasets using the Huggingface API, but you also have a local version of Unitxt
installed, and the versions are not compatible.  You should either update the local installed Unitxt
to the Unitxt Huggingface dataset version, or uninstall the local Unitxt package (in case you don't require the access to Unitxt
direct APIs), or change the code to load the datasets using the direct Unitxt APIs and not use the Huggingface API.

