.. _examples:

========
Examples
========

Here you will find complete coding samples showing how to perform different tasks using Unitxt.
Each example comes with a self contained python file that you can run and later modify.


Basic Usage
------------

Evaluate an existing dataset from the Unitxt catalog. 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Demonstrates how to evaluate an existing entailment dataset using Unitxt.
Unitxt is used to load the dataset, generate the input to the model, run inference and evaluate the results.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_existing_dataset_with_install.py>`__

Related documentation: :ref:`Installation <install_unitxt>` , :ref:`WNLI dataset card in catalog <catalog.cards.wnli>`, :ref:`Relation template in catalog <catalog.templates.classification.multi_class.relation.default>`, :ref:`Inference Engines <inference>`.


Evaluate a custom dataset
+++++++++++++++++++++++++

This example demonstrates how to evaluate a user QA answering dataset in a standalone file using a user-defined task and template.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/standalone_qa_evaluation.py>`__

Related documentation: :ref:`Add new dataset tutorial <adding_dataset>`.

Evaluate a custom dataset - reusing existing catalog assets
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example demonstrates how to evaluate a user QA dataset using the predefined open qa task and templates.
It also shows how to use preprocessing steps to align the raw input of the dataset with the predefined task fields.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/qa_evaluation.py>`__

Related documentation: :ref:`Add new dataset tutorial <adding_dataset>`, :ref:`Open QA task in catalog <catalog.tasks.qa.open>`, :ref:`Open QA template in catalog <catalog.templates.qa.open.title>`, :ref:`Inference Engines <inference>`.

Evaluate a custom dataset - with existing predictions
=====================================================

These examples demonstrate how to evaluate a datasets of different tasks when predictions are already available and no inference is required.

`Example code for QA task  <https://github.com/IBM/unitxt/blob/main/examples/evaluate_qa_dataset_with_given_predictions.py>`__

`Example code for classification task  <https://github.com/IBM/unitxt/blob/main/examples/evaluate_classification_dataset_with_given_predictions.py>`__  

Related documentation: :ref:`Evaluating datasets <evaluating_datasets>`

Evaluate a Named Entity Recognition (NER) dataset
===================================================

This example demonstrates how to evaluate a named entity recognition task.
The ground truth entities are provided as spans within the provided texts, 
and the model is prompted to identify these entities.
Classifical f1_micro, f1_macro, and per-entity-type f1 metrics are reported.

`Example code  <https://github.com/IBM/unitxt/blob/main/examples/ner_evaluation.py>`__

Related documentation: :ref:`Add new dataset tutorial <adding_dataset>`, :ref:`Open NER task in catalog <catalog.tasks.ner.all_entity_types>`, :ref:`Inference Engines <inference>`.

Evaluation usecases
-----------------------

Evaluate the impact of different templates and in-context learning demonstrations
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example demonstrates how different templates and the number of in-context learning examples impacts the performance of a model on an entailment task.
It also shows how to register assets into a local catalog and reuse them.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_different_templates.py>`__

Related documentation: :ref:`Templates tutorial <adding_template>`, :ref:`Formatting tutorial <adding_format>`, :ref:`Using the Catalog <using_catalog>`, :ref:`Inference Engines <inference>`.

Evaluate the impact of different formats and system prompts
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example demonstrates how different formats and system prompts affect the input provided to a llama3 chat model and evaluate their impact on the obtained scores.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_different_formats.py>`__

Related documentation: :ref:`Formatting tutorial <adding_format>`.

Evaluate the impact of different demonstration example selections
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example demonstrates how different methods of selecting the demonstrations in in-context learning affect the results.
Three methods are considered: fixed selection of example demonstrations for all test instances,
random selection of example demonstrations for each test instance,
and choosing the demonstration examples most (lexically) similar to each test instance.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_different_demo_selections.py>`__

Related documentation: :ref:`Formatting tutorial <adding_format>`.

Evaluate dataset with a pool of templates and some number of demonstrations
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example demonstrates how to evaluate a dataset using a pool of templates and a varying number of in-context learning demonstrations. It shows how to sample a template and specify the number of demonstrations for each instance from predefined lists.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_different_templates_num_demos.py>`__

Related documentation: :ref:`Templates tutorial <adding_template>`, :ref:`Formatting tutorial <adding_format>`, :ref:`Using the Catalog <using_catalog>`, :ref:`Inference Engines <inference>`.

Long Context
+++++++++++++++++++++++++++++

This example explores the effect of long context in classification.
It converts a standard multi class classification dataset (sst2 sentiment classification),
where single sentence texts are classified one by one, to a dataset
where multiple sentences are classified using a single LLM call.
It compares the f1_micro in both approaches on two models.
It uses serializers to verbalize and enumerated list of multiple sentences and labels.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_batched_multiclass_classification.py>`__

Related documentation:  :ref:`Sst2 dataset card in catalog <catalog.cards.sst2>` :ref:`Types and Serializers Guide <types_and_serializers>`.

Construct a benchmark of multiple datasets and obtain the final score
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example shows how to construct a benchmark that includes multiple datasets, each with a specific template. It demonstrates how to use these templates to evaluate the datasets and aggregate the results to obtain a final score. This approach provides a comprehensive evaluation across different tasks and datasets.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_benchmark.py>`__

Related documentation: :ref:`Benchmarks tutorial <adding_benchmark>`, :ref:`Formatting tutorial <adding_format>`, :ref:`Using the Catalog <using_catalog>`, :ref:`Inference Engines <inference>`.

LLM as Judges
--------------

Using LLM as judge for direct comparison using a predefined criteria
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example demonstrates how to use LLM-as-a-Judge with a predefined criteria, in this case *answer_relevance*. The unitxt catalog has more than 40 predefined criteria for direct evaluators.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_llm_as_judge_direct_predefined_criteria.py>`__

Related documentation: :ref:`Using LLM as a Judge in Unitxt <llm_as_judge>`


Using LLM as judge for direct comparison using a custom criteria
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The user can also specify a bespoke criteria that the judge model uses as a guide to evaluate the responses.
This example demonstrates how to use LLM-as-a-Judge with a user-defined criteria. The criteria must have options and option_map.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_llm_as_judge_direct_user_criteria_no_catalog.py>`__

Related documentation: :ref:`Creating a custom criteria`


Evaluate an existing dataset using an LLM-as-a-Judge for direct comparison
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example demonstrates how to evaluate an existing QA dataset (squad) using the HuggingFace Datasets and Evaluate APIs and leveraging a predefined criteria for direct evaluation.
Note that here we also showcase unitxt's ability to evaluate the dataset on multiple criteria, namely, *answer_relevance*, *coherence* and *conciseness*

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_existing_dataset_by_llm_as_judge_direct.py>`__

Related documentation: :ref:`End to end Direct example`


Using LLM as a judge for pairwise comparison using a predefined criteria
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example demonstrates how to use LLM-as-a-Judge for pairwise comparison using a predefined criteria from the catalog. The unitxt catalog has 7 predefined criteria for pairwise evaluators.
We also showcase that the criteria does not need to be the same across the entire dataset and that the framework can handle different criteria for each datapoint.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_llm_as_judge_pairwise_predefined_criteria.py>`__

This example demonstrates using LLM-as-a-Judge for pairwise comparison using a single predefined criteria for the entire dataset

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_llm_as_judge_pairwise_criteria_from_dataset.py>`__


Evaluate an existing dataset using an LLM-as-a-Judge for direct comparison
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example demonstrates how to evaluate an existing QA dataset (squad) using the HuggingFace Datasets and Evaluate APIs and leveraging a predefined criteria for pairwise evaluation.
Note that here we also showcase unitxt's ability to evaluate the dataset on multiple criteria, namely, *answer_relevance*, *coherence* and *conciseness*

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_existing_dataset_by_llm_as_judge_direct.py>`__

Related documentation: :ref:`End to end Pairwise example`


RAG
---

Evaluate RAG response generation
++++++++++++++++++++++++++++++++

This example demonstrates how to use the standard Unitxt RAG response generation task.
The response generation task is the following:
Given a question and one or more context(s), generate an answer that is correct and faithful to the context(s).
The example shows how to map the dataset input fields to the RAG response task fields
and use the existing metrics to evaluate model results.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_rag_response_generation.py>`__

Related documentation: :ref:`RAG Guide <rag_support>`, :ref:`Response generation task <catalog.tasks.rag.response_generation>`, :ref:`Inference Engines <inference>`.

Evaluate RAG End to End - with existing predictions
=====================================================

This example demonstrates how to evaluate an end to end RAG system, given that the RAG system outputs are available.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_rag_end_to_end_dataset_with_given_predictions.py>`__

Related documentation: :ref:`Evaluating datasets <evaluating_datasets>`

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

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_image_text_to_text.py>`__

Related documentation: :ref:`Multi-Modality Guide <multi_modality>`, :ref:`Inference Engines <inference>`.


Evaluate Image-Text to Text Model With Different Templates
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Evaluate Image-Text to Text Models with different templates and explore the sensitivity of the model to different textual variations.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_image_text_to_text_with_different_templates.py>`__

Related documentation: :ref:`Multi-Modality Guide <multi_modality>`, :ref:`Inference Engines <inference>`.

Advanced topics
----------------------------

Custom Types and Serializers
+++++++++++++++++++++++++++++

This example show how to define new data types as well as the way these data type should be handled when processed to text.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/custom_types.py>`__

Related documentation: :ref:`Types and Serializers Guide <types_and_serializers>`, :ref:`Inference Engines <inference>`.


Evaluate an existing dataset from the Unitxt catalog (No installation)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example demonstrates how to evaluate an existing entailment dataset (wnli) using HuggingFace Datasets and Evaluate APIs, with no installation required.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_existing_dataset_no_install.py>`__

Related documentation:  :ref:`Evaluating datasets <evaluating_datasets>`, :ref:`WNLI dataset card in catalog <catalog.cards.wnli>`, :ref:`Relation template in catalog <catalog.templates.classification.multi_class.relation.default>`, :ref:`Inference Engines <inference>`.
