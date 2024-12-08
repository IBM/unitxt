.. _inference:

==============
Inference
==============

.. note::

   This tutorial requires a :ref:`Unitxt installation <install_unitxt>`.

Introduction
------------
Unitxt offers a wide array of :class:`Inference Engines <unitxt.inference>` for running models either locally (using HuggingFace, Ollama, and VLLM) or by making API requests to services like WatsonX, OpenAI, AWS, and Together AI.

Unitxt inference engines serve two main purposes:

    1. Running a full end-to-end evaluation pipeline with inference.
    2. Using models for intermediate steps, such as evaluating other models (e.g., LLMs as judges) or for data augmentation.

Running Models Locally
-----------------------
You can run models locally with inference engines like:

    - :class:`HFPipelineBasedInferenceEngine <unitxt.inference.HFPipelineBasedInferenceEngine>`
    - :class:`VLLMInferenceEngine <unitxt.inference.VLLMInferenceEngine>`
    - :class:`OllamaInferenceEngine <unitxt.inference.OllamaInferenceEngine>`

To get started, prepare your engine:

.. code-block:: python

    engine = HFPipelineBasedInferenceEngine(
        model_name="meta-llama/Llama-3.2-1B", max_new_tokens=32
    )

Then load the data:

.. code-block:: python

    dataset = load_dataset(
        card="cards.xsum",
        template="templates.summarization.abstractive.formal",
        format="formats.chat_api",
        metrics=[llm_judge_with_summary_metric],
        loader_limit=5,
        split="test",
    )

Notice: we create the data with  `format="formats.chat_api"` which produce data as list of chat turns:

.. code-block:: python

    [
        {"role": "system", "content": "Summarize the following Document."},
        {"role": "user", "content": "Document: <...>"}
    ]

Now run inference on the dataset:

.. code-block:: python

    predictions = engine.infer(dataset)

Finally, evaluate the predictions and obtain final scores:

.. code-block:: python

    evaluate(predictions=predictions, data=dataset)

Calling Models Through APIs
---------------------------
Calling models through an API is even simpler and is primarily done using one class: :class:`CrossProviderInferenceEngine <unitxt.inference.CrossProviderInferenceEngine>`.

You can create a :class:`CrossProviderInferenceEngine` as follows:

.. code-block:: python

    engine = CrossProviderInferenceEngine(
        model="llama-3-2-1b-instruct", provider="watsonx"
    )

This engine supports providers such as ``watsonx``, ``together-ai``, ``open-ai``, ``aws``, ``ollama``, ``bam``, and ``watsonx-sdk``.

It can be used with all supported models listed here: :class:`supported models <unitxt.inference.CrossProviderInferenceEngine>`.

Running inference follows the same pattern as before:

.. code-block:: python

    predictions = engine.infer(dataset)

Creating a Cross-API Engine
---------------------------
Alternatively, you can create an engine without specifying a provider:

.. code-block:: python

    engine = CrossProviderInferenceEngine(
        model="llama-3-2-1b-instruct"
    )

You can set the provider later by:

.. code-block:: python

    import unitxt

    unitxt.settings.default_provider = "watsonx"

Or by setting an environment variable:

.. code-block:: bash

    export UNITXT_DEFAULT_PROVIDER="watsonx"