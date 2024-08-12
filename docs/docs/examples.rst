.. _examples:
==============
Examples ✨
==============

Here you find complete examples showing how to perform different tasks using Unitxt.
Each example is a self contained python file that you can run and later modify.


Basic Usage
------------

Evaluate an existing dataset from the Unitxt catalog
++++++++++++++++++++++++++++++++++++++++++++++++++++

Demonstrates how to evaluate an existing entailment dataset (wnli) using Huggingface  datasets and evaluate APIs, with no installation required.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_existing_dataset_no_install.py>`_

Related documentation:  :ref:`Evaluating datasets <evaluating_datasets>`, :ref:`WNLI dataset card in catalog <catalog.cards.wnli>`, :ref:`Relation template in catalog <catalog.templates.classification.multi_class.relation.default>`.

Evaluate a custom dataset
+++++++++++++++++++++++++

Demonstrates how to evaluate a user QA answering dataset in a standalone file using a user defined task and template.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/standalone_qa_evaluation.py>`_

Related documentation: :ref:`Add new dataset tutorial <adding_dataset>`.

Evaluate a custom dataset - reusing existing catalog assets
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Demonstrates how to evaluate a user QA dataset using the predefined open qa task and templates.
It also shows how to use preprocessing steps to align the raw input of the dataset with the predefined task fields.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/qa_evaluation.py>`_

Related documentation: :ref:`Add new dataset tutorial <adding_dataset>`, :ref:`Open QA task in catalog <catalog.tasks.qa.open>`, :ref:`Open QA template in catalog <catalog.templates.qa.open.title>`.

Evaluate the impact of different templates and in-context learning demonstrations
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Demonstrates how different templates and number of in-context learning examples impacts performance of a model on an entailment task.
It also shows how to register assets into a local catalog and reuse them.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_different_templates.py>`_

Related documentation: :ref:`Templates tutorial <adding_template>`, :ref:`Formatting tutorial <adding_format>`, :ref:`Using the Catalog <using_catalog>`.

Evaluate the impact of different formats and system prompts
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Demonstrates how different formats and system prompts effect the input provided to a llama3 chat model and evaluate their impact on the obtain scores.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_different_formats.py>`_

Related documentation: :ref:`Formatting tutorial <adding_format>`.

Evaluate the impact of different demonstration example selections
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Demonstrates how different methods of selecting the demonstrations in in-context learning affect the results.
Three methods are considered: fixed selection of example demonstrations for all test instance,
random selection of example demonstrations for each test instance,
and choosing the demonstration examples most (lexically) similar to each test instance.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_different_demo_selections.py>`_

Related documentation: :ref:`Formatting tutorial <adding_format>`.

Evaluate dataset with a pool of templates and number of demonstrations
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Demonstrates how to evaluate a dataset using a pool of templates and a varying number of in-context learning demonstrations. It shows how to sample a template and the number of demonstrations for each instance from predefined lists.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_different_templates_num_demos.py>`_

Related documentation: :ref:`Templates tutorial <adding_template>`, :ref:`Formatting tutorial <adding_format>`, :ref:`Using the Catalog <using_catalog>`.

LLM as Judges
--------------

Evaluate an existing dataset using a pre-defined LLM as judge
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Demonstrates how to evaluate an existing QA dataset (squad) using the Huggingface datasets and evaluate APIs and leveraging a predefine LLM as a judge metric.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_dataset_by_llm_as_judge_no_install.py>`_

Related documentation: :ref:`Evaluating datasets <evaluating_datasets>`, :ref:`LLM as a Judge Metrics Guide <llm_as_judge>`.

Evaluate a custom dataset using a custom LLM as Judge
+++++++++++++++++++++++++++++++++++++++++++++++++++++

Demonstrates how to evaluate a user QA answering dataset in a standalone file using a user defined task and template. In addition, it shows how to define an LLM as a judge metric, specify the template it uses to produce the input to the judge, and select the judge model and platform.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/standalone_evaluation_llm_as_judge.py>`_

Related documentation: :ref:`LLM as a Judge Metrics Guide <llm_as_judge>`.

Evaluate an existing dataset from the catalog comparing two custom LLM as judges
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Demonstrates how to evaluate a document summarization dataset by defining an LLM as a judge metric, specifying the template it uses to produce the input to the judge, and selecting the judge model and platform.
The example adds two LLM judges, one that uses the ground truth (references) from the dataset and one that does not.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_summarization_dataset_llm_as_judge.py>`_

Related documentation: :ref:`LLM as a Judge Metrics Guide <llm_as_judge>`.

Evaluate the quality of an LLM as judge
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Demonstrates how to evaluate an LLM as judge by checking its scores using the gold references of a dataset.
It checks if the judge consistently prefers correct outputs over clearly wrong ones.
Note that to check the the ability of the LLM as judge to discern suitable differences between
partially correct answers requires more refined tests and corresponding labeled data.
The example shows an 8b llama based judge is not a good judge for a summarization task,
while the 70b model performs much better.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_llm_as_judge.py>`_

Related documentation: :ref:`LLM as a Judge Metrics Guide <llm_as_judge>`.


Evaluate your model on the Arena Hard benchmark using a custom LLMaJ
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Demonstrates how to evaluate a user model on the Arena Hard benchmark, using an LLMaJ other than the GPT4.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_a_model_using_arena_hard.py>`_

Related documentation: :ref:`Evaluate a Model on Arena Hard Benchmark <arena_hard_evaluation>`.

Evaluate a judge model performance judging the Arena Hard Benchmark
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Demonstrates how to evaluate the capabilities of a user model, to act as a judge on the Arena Hard benchmark.
The model is evaluated on its capability to give a judgment that is in correlation with GPT4 judgment on the benchmark.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_a_judge_model_capabilities_on_arena_hard.py>`_

Related documentation: :ref:`Evaluate a Model on Arena Hard Benchmark <arena_hard_evaluation>`.

Evaluate using ensemble of LLM as a judge metrics
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Demonstrates how to create a metric which is an ensemble of LLM as a judge metrics.
The example shows how to ensemble two judges which uses different templates.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_using_metrics_ensemble.py>`_

Related documentation: :ref:`LLM as a Judge Metrics Guide <llm_as_judge>`.


RAG
---

Evaluate RAG response generation
++++++++++++++++++++++++++++++++

Demonstrates how to use the standard Unitxt RAG response generation task.
The response generation task is the following:
Given a question and one or more context, generate an answer that is correct and faithful to the contexts.
The example shows how to map the dataset input fields to the RAG response task fields
and use the existing metrics to evaluate model results.

`Example code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_rag_response_generation.py>`_

Related documentation: :ref:`RAG Guide <rag_support>`.  :ref:`Response generation task <catalog.tasks.rag.response_generation>`.

