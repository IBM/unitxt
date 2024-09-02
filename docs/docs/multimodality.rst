.. _multi_modality:

==============
Multi-Modality
==============

.. note::

   This tutorial requires a :ref:`Unitxt installation <install_unitxt>`.

Introduction
------------

This tutorial explores multi-modality processing with Unitxt, focusing on handling image-text-to-text datasets and creating an evaluation and inference pipeline. By the end, you'll be equipped to process complex multi-modal data efficiently.

Part 1: Understanding Image-Text to Text Tasks
----------------------------------------------

Image-text to text tasks involve providing a model with a combination of text and images and expecting a textual answer. These tasks are increasingly relevant in modern AI applications.

Tutorial Overview
^^^^^^^^^^^^^^^^^

We'll create an image-text to text evaluation pipeline using Unitxt, concentrating on a document visual question answering (DocVQA) task. This task involves asking questions about images and generating textual answers.

Part 2: Data Preparation
------------------------

Creating a Unitxt DataCard
^^^^^^^^^^^^^^^^^^^^^^^^^^

Our first step is to prepare the data using a Unitxt DataCard. If you it's your first time adding a DataCard we recommend reading the :ref:`Adding Datasets Tutorial <adding_dataset>`.

Dataset Selection
^^^^^^^^^^^^^^^^^

We'll use the ``doc_vqa`` dataset from Hugging Face, formatting it for a question-answering task. Specifically, we'll use the ``tasks.qa.with_context.abstractive`` task from the Unitxt Catalog.

DataCard Implementation
^^^^^^^^^^^^^^^^^^^^^^^

Our goal in the DataCard will be to adjust the data as it comes from hugginface to task schema.
Create a Python file named ``doc_vqa.py`` and implement the DataCard as follows:

.. code-block:: python

    from unitxt.blocks import LoadHF, Set, TaskCard
    from unitxt.collections_operators import Explode, Wrap
    from unitxt.image_operators import ImageToText
    from unitxt.operators import Copy

    card = TaskCard(
        loader=LoadHF(path="cmarkea/doc-vqa"),
        preprocess_steps=[
            "splitters.small_no_dev",
            Explode(field=f"qa/en", to_field="pair"),
            Copy(field="pair/question", to_field="question"),
            Copy(field="pair/answer", to_field="answers"),
            Wrap(field="answers", inside="list"),
            Set(fields={"context_type": "image"}),
            ImageToText(field="image", to_field="context"),
        ],
        task="tasks.qa.with_context.abstractive",
        templates="templates.qa.with_context.all",
    )

The ImageToText Operator
^^^^^^^^^^^^^^^^^^^^^^^^

The ``ImageToText`` operator is a key component that integrates the image into the text, allowing inference engines to process both elements simultaneously.

Testing and Catalog Addition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Test the card and add it to the catalog:

.. code-block:: python

    test_card(card)
    add_to_catalog(card, f"cards.doc_vqa.en", overwrite=True)

Part 3: Inference and Evaluation
--------------------------------

With our data prepared, we can now test model performance.

Pipeline Setup
^^^^^^^^^^^^^^

Set up the inference and evaluation pipeline:

.. code-block:: python

    from unitxt.api import evaluate, load_dataset
    from unitxt.inference_engines import HFLlavaInferenceEngine
    from unitxt.text_utils import print_dict

    # Initialize the inference model
    inference_model = HFLlavaInferenceEngine(
        model_name="llava-hf/llava-interleave-qwen-0.5b-hf", max_new_tokens=32
    )

    # Load and prepare the dataset
    dataset = load_dataset(
        card="cards.doc_vqa.en",
        template="templates.qa.with_context.title",
        format="formats.models.llava_interleave",
        loader_limit=30,
    )

    # Select a subset for testing
    test_dataset = dataset["test"].select(range(5))

Executing Inference and Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the model and evaluate the results:

.. code-block:: python

    # Perform inference
    predictions = inference_model.infer(test_dataset)

    # Evaluate the predictions
    evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

    # Print the results
    print_dict(
        evaluated_dataset[0],
        keys_to_print=[
            "source",
            "media",
            "references",
            "processed_prediction",
            "score"
        ],
    )

Conclusion
----------

You have now successfully implemented an image-text to text evaluation pipeline with Unitxt. This tool enables the processing of complex multi-modal data, opening up new possibilities for AI applications.

We encourage you to explore further by experimenting with different datasets, models, and tasks to fully leverage Unitxt's capabilities in multi-modal processing.