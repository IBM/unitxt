.. title:: Unitxt Embraces Rich Chat Format and Cross API Inference: Simplifying LLM Evaluation

:Authors: Elron Bandel

:Date: 2024-11-19

=================================================================================================
[19/11/2024] Unitxt Embraces Rich Chat Format and Cross API Inference: Simplifying LLM Evaluation
=================================================================================================

**Authors**: Elron Bandel

``19/11/2024``

Preparing data for training and testing language models is a complex task.
It involves handling various data formats, preprocessing, and ways of verbalizing tasks.
Ensuring reproducibility and compatibility across platforms further adds to the complexity.

Recognizing these challenges, Unitxt has always aimed to simplify data preparation.
Today, we are introducing two major updates to redefine our support for LLM workflows.

Introducing Two Major Enhancements
-----------------------------------

1. **Producing Data in Chat API Format**
   Unitxt now can produces data in the widely adopted Chat API format.
   This ensures compatibility with popular LLM Provider APIs and avoid the need from custom per model formatting.
   Additionally, the format supports multiple modalities such as text, images, and videos.

2. **A Comprehensive Array of Inference Engines**
   We added wrappers for local inference platforms like Llama and Hugging Face
   as well as remote APIs such as LiteLLM, OpenAI, Watsonx, and more.
   
   These wrappers make executing evaluation and inference tasks seamless
   and platform-agnostic, in just a `few lines of code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_existing_dataset_with_install.py>`_.


.. code-block:: python

    # Illustration of rich chat api ready for inference:

    [
        {
            "role": "system",
            "content": "You are an assistant that helps classify images."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What does this image depict?"
                },
                {
                    "type": "image",
                    "image": {
                        "mime_type": "image/jpeg",
                        "data": <ENCODED_IMAGE>
                    }
                }
            ]
        }
    ]

Expanding Opportunities for the Community
------------------------------------------

These updates unlock significant opportunities, including:

- **Full Evaluation Pipelines**:
  Design and execute `end-to-end workflows <https://www.unitxt.ai/en/latest/docs/examples.html#evaluation-usecases>`_ directly in the Unitxt framework.
  For example, evaluate the impact of different templates, in-context example selection, answering multiple questions in one inference, and more.

- **Multi-Modality Evaluation**:
  Evaluate `models with diverse inputs <https://www.unitxt.ai/en/latest/docs/examples.html#multi-modality>`_, from text to images and beyond.

- **Easy Assembly of LLM Judges**:
  Quickly set up `LLMs as evaluators <https://github.com/IBM/unitxt/blob/main/examples/standalone_evaluation_llm_as_judge.py>`_ using Unitxt inference engines.


Our Commitment to Collaboration
-------------------------------

Although you can now run end to end evaluation in Unitxt, Unitxt is still a general data preparation library.
That means we remain committed to partnerships with other evaluation platforms such as `HELM <https://www.unitxt.ai/en/latest/docs/helm.html>`_, `LM Eval Harness <https://www.unitxt.ai/en/latest/docs/lm_eval.html>`_, and others.
Our Chat API format and inference engine support enhance accessibility and compatibility.
These updates empower our partners to adopt the latest standards seamlessly.

Conclusion
----------

Unitxt is adapting to the evolving landscape of language models and their capabilities.
By supporting the Chat API format and inference engines, we simplify model workflows.
These updates position Unitxt as the premier platform for LLM evaluation and integration.

We invite you to explore these features and join us in advancing model capabilities.

---

For more information, visit the :ref:`inference engines guide <inference>` or see many of our :ref:`code examples <examples>`.
