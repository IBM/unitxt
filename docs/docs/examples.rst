.. _examples:
==============
Examples ✨
==============

Here you will find complete coding samples showing how to perform different tasks using Unitxt.
Each example comes with a self contained python file that you can run and later modify.


Basic Usage
------------


Evaluate an existing dataset from the Unitxt catalog (No installation)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example demonstrates how to evaluate an existing entailment dataset (wnli) using HuggingFace Datasets and Evaluate APIs, with no installation required.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_existing_dataset_no_install.py>`_

Related documentation:  :ref:`Evaluating datasets <evaluating_datasets>`, :ref:`WNLI dataset card in catalog <catalog.cards.wnli>`, :ref:`Relation template in catalog <catalog.templates.classification.multi_class.relation.default>`.

Evaluate an existing dataset from the Unitxt catalog (with Unitxt installation)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Demonstrates how to evaluate an existing entailment dataset (wnli) using Unitxt native APIs.
This approach is faster than using Huggingface APIs.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_existing_dataset_with_install.py>`_

Related documentation: :ref:`Installation <installation>` , :ref:`WNLI dataset card in catalog <catalog.cards.wnli>`, :ref:`Relation template in catalog <catalog.templates.classification.multi_class.relation.default>`.


Evaluate a custom dataset
+++++++++++++++++++++++++

This example demonstrates how to evaluate a user QA answering dataset in a standalone file using a user-defined task and template.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/standalone_qa_evaluation.py>`_

Related documentation: :ref:`Add new dataset tutorial <adding_dataset>`.

Evaluate a custom dataset - reusing existing catalog assets
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example demonstrates how to evaluate a user QA dataset using the predefined open qa task and templates.
It also shows how to use preprocessing steps to align the raw input of the dataset with the predefined task fields.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/qa_evaluation.py>`_

Related documentation: :ref:`Add new dataset tutorial <adding_dataset>`, :ref:`Open QA task in catalog <catalog.tasks.qa.open>`, :ref:`Open QA template in catalog <catalog.templates.qa.open.title>`.

Evaluate the impact of different templates and in-context learning demonstrations
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example demonstrates how different templates and the number of in-context learning examples impacts the performance of a model on an entailment task.
It also shows how to register assets into a local catalog and reuse them.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_different_templates.py>`_

Related documentation: :ref:`Templates tutorial <adding_template>`, :ref:`Formatting tutorial <adding_format>`, :ref:`Using the Catalog <using_catalog>`.

Evaluate the impact of different formats and system prompts
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example demonstrates how different formats and system prompts affect the input provided to a llama3 chat model and evaluate their impact on the obtained scores.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_different_formats.py>`_

Related documentation: :ref:`Formatting tutorial <adding_format>`.

Evaluate the impact of different demonstration example selections
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example demonstrates how different methods of selecting the demonstrations in in-context learning affect the results.
Three methods are considered: fixed selection of example demonstrations for all test instances,
random selection of example demonstrations for each test instance,
and choosing the demonstration examples most (lexically) similar to each test instance.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_different_demo_selections.py>`_

Related documentation: :ref:`Formatting tutorial <adding_format>`.

Evaluate dataset with a pool of templates and some number of demonstrations
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example demonstrates how to evaluate a dataset using a pool of templates and a varying number of in-context learning demonstrations. It shows how to sample a template and specify the number of demonstrations for each instance from predefined lists.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_different_templates_num_demos.py>`_

Related documentation: :ref:`Templates tutorial <adding_template>`, :ref:`Formatting tutorial <adding_format>`, :ref:`Using the Catalog <using_catalog>`.

Construct a benchmark of multiple datasets and obtain the final score
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example shows how to construct a benchmark that includes multiple datasets, each with a specific template. It demonstrates how to use these templates to evaluate the datasets and aggregate the results to obtain a final score. This approach provides a comprehensive evaluation across different tasks and datasets.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_benchmark.py>`_

Related documentation: :ref:`Benchmarks tutorial <adding_benchmark>`, :ref:`Formatting tutorial <adding_format>`, :ref:`Using the Catalog <using_catalog>`.

LLM as Judges
--------------

Evaluate an existing dataset using a predefined LLM as judge
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example demonstrates how to evaluate an existing QA dataset (squad) using the HuggingFace Datasets and Evaluate APIs and leveraging a predefine LLM as a judge metric.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_existing_dataset_by_llm_as_judge.py>`_

Related documentation: :ref:`Evaluating datasets <evaluating_datasets>`, :ref:`LLM as a Judge Metrics Guide <llm_as_judge>`.

Evaluate a custom dataset using a custom LLM as Judge
+++++++++++++++++++++++++++++++++++++++++++++++++++++

This example demonstrates how to evaluate a user QA answering dataset in a standalone file using a user-defined task and template. In addition, it shows how to define an LLM as a judge metric, specify the template it uses to produce the input to the judge, and select the judge model and platform.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/standalone_evaluation_llm_as_judge.py>`_

Related documentation: :ref:`LLM as a Judge Metrics Guide <llm_as_judge>`.

Evaluate an existing dataset from the catalog comparing two custom LLM as judges
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example demonstrates how to evaluate a document summarization dataset by defining an LLM as a judge metric, specifying the template it uses to produce the input to the judge, and selecting the judge model and platform.
The example adds two LLM judges, one that uses the ground truth (references) from the dataset and one that does not.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_summarization_dataset_llm_as_judge.py>`_

Related documentation: :ref:`LLM as a Judge Metrics Guide <llm_as_judge>`.

Evaluate the quality of an LLM as judge
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example demonstrates how to evaluate an LLM as judge by checking its scores using the gold references of a dataset.
It checks if the judge consistently prefers correct outputs over clearly wrong ones.
Note that to check the the ability of the LLM as judge to discern suitable differences between
partially correct answers requires more refined tests and corresponding labeled data.
The example shows an 8b llama based judge is not a good judge for a summarization task,
while the 70b model performs much better.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_llm_as_judge.py>`_

Related documentation: :ref:`LLM as a Judge Metrics Guide <llm_as_judge>`.


Evaluate your model on the Arena Hard benchmark using a custom LLMaJ
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example demonstrates how to evaluate a user model on the Arena Hard benchmark, using an LLMaJ other than the GPT4.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_a_model_using_arena_hard.py>`_

Related documentation: :ref:`Evaluate a Model on Arena Hard Benchmark <arena_hard_evaluation>`.

Evaluate a judge model performance judging the Arena Hard Benchmark
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example demonstrates how to evaluate the capabilities of a user model, to act as a judge on the Arena Hard benchmark.
The model is evaluated on its capability to give a judgment that is in correlation with GPT4 judgment on the benchmark.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_a_judge_model_capabilities_on_arena_hard.py>`_

Related documentation: :ref:`Evaluate a Model on Arena Hard Benchmark <arena_hard_evaluation>`.

Evaluate using ensemble of LLM as a judge metrics
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example demonstrates how to create a metric which is an ensemble of LLM as a judge metrics.
The example shows how to ensemble two judges which uses different templates.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_using_metrics_ensemble.py>`_

Related documentation: :ref:`LLM as a Judge Metrics Guide <llm_as_judge>`.


RAG
---

Evaluate RAG response generation
++++++++++++++++++++++++++++++++

This example demonstrates how to use the standard Unitxt RAG response generation task.
The response generation task is the following:
Given a question and one or more context(s), generate an answer that is correct and faithful to the context(s).
The example shows how to map the dataset input fields to the RAG response task fields
and use the existing metrics to evaluate model results.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_rag_response_generation.py>`_

Related documentation: :ref:`RAG Guide <rag_support>`.  :ref:`Response generation task <catalog.tasks.rag.response_generation>`.

Multi-Modality
--------------

Evaluate Image-Text to Text Model
+++++++++++++++++++++++++++++++++
This example demonstrates how to evaluate an image-text to text model using Unitxt.
The task involves generating text responses based on both image and text inputs. This is particularly useful for tasks like visual question answering (VQA) where the model needs to understand and reason about visual content to answer questions.
The example shows how to:

    1. Load a pre-trained image-text model (LLaVA in this case)
    2. Prepare a dataset with image-text inputs
    3. Run inference on the model
    4. Evaluate the model's predictions

The code uses the document VQA dataset in English, applies a QA template with context, and formats it for the LLaVA model. It then selects a subset of the test data, generates predictions, and evaluates the results.
This approach can be adapted for various image-text to text tasks, such as image captioning, visual reasoning, or multimodal dialogue systems.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_image_text_to_text.py>`_

Related documentation: :ref:`Multi-Modality Guide <multi_modality>`.
