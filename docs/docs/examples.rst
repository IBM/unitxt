.. _examples:
==============
Examples ✨
==============

Here you find complete examples showing how to perform different tasks using Unitxt. 
Each example is a self contained python file that you can run and later modify.


.. list-table:: Common Usecases
   :widths: 50 50 50 50
   :header-rows: 1

   * - What do you want to do?
     - Description
     - Link to code
     - Related documentation
   * - Evaluate an existing dataset from the Unitxt catalog
     - Demonstrates how to evaluate an existing entailment dataset (wnli) using Huggingface 
       datasets and evaluate APIs, with no installation required.  
     - `code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_existing_dataset_no_install.py>`_
     - | :ref:`Evaluating datasets <evaluating_datasets>`.  
       | :ref:`WNLI dataset card in catalog <catalog.cards.wnli>`.
       | :ref:`Relation template in catalog <catalog.templates.classification.multi_class.relation.default>`.
   * - Evaluate your question-answering dataset 
     - Demonstrates how to evaluate a user QA answering dataset in a standalone file using a user defined task and template.
     - `code <https://github.com/IBM/unitxt/blob/main/examples/standalone_qa_evaluation.py>`_
     - :ref:`Add new dataset tutorial <adding_dataset>`.
   * - Evaluate your question-answering dataset  - reusing existing catalog assets
     - Demonstrates how to evaluate a user QA dataset using the predefined open qa task and templates.
       It also shows how to use preprocessing steps to align the raw input of the dataset with the predefined task fields.
     - `code <https://github.com/IBM/unitxt/blob/main/examples/qa_evaluation.py>`_
     - | :ref:`Add new dataset tutorial <adding_dataset>`.  
       | :ref:`Open QA task in catalog <catalog.tasks.qa.open>`.
       | :ref:`Open QA template in catalog <catalog.templates.qa.open.title>`.
   * - Evaluate the impact of different formats and system prompts on the same task
     - Demonstrates how different formats and system prompts effect the input provided to a llama3 chat model and evaluate their impact on the obtain scores.
     - `code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_different_formats.py>`_
     - | :ref:`Formatting tutorial <adding_format>`.



.. list-table:: LLM as a judge
   :widths: 50 50 50 50
   :header-rows: 1

   * - What do you want to do?
     - Description
     - Link to code
     - Related documentation
   * - Evaluate your question-answering dataset  
     - Demonstrates how to evaluate a user QA answering dataset in a standalone file using a user defined task and template. In addition, it shows how to define an LLM as a judge metric, specify the template it uses to produce the input to the judge, and select the judge model and platform.
     - `code <https://github.com/IBM/unitxt/blob/main/examples/standalone_evaluation_llm_as_judge.py>`_
     - | :ref:`LLM as a Judge Metrics Guide <llm_as_judge>`.
   * - Evaluate an existing summarization dataset from the catalog with LLM as judge
     - Demonstrates how to evaluate a document summarization dataset by define an LLM as a judge metric, specify the template it uses to produce the input to the judge, and select the judge model and platform.
     - `code <https://github.com/IBM/unitxt/blob/main/examples/evaluation_summarization_dataset_llm_as_judge>`_
     - | :ref:`LLM as a Judge Metrics Guide <llm_as_judge>`.
   * - Evaluate your model on the Arena Hard benchmark using a custom LLMaJ.
     - Demonstrates how to evaluate a user model on the Arena Hard benchmark, using an LLMaJ other than the GPT4.
     - `code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_a_model_using_arena_hard>`_
     - | :ref:`Evaluate a Model on Arena Hard Benchmark <arena_hard_evaluation>`.
   * - Evaluate a judge model performance judging the Arena Hard Benchmark.
     - Demonstrates how to evaluate the capabilities of a user model, to act as a judge on the Arena Hard benchmark. The model is evaluated on it's capabilities to give a judgment that is in correlation with GPT4 judgment on the benchmark.
     - `code <https://github.com/IBM/unitxt/blob/main/examples/evaluate_a_judge_model_capabilities_on_arena_hard>`_
     - | :ref:`Evaluate a Model on Arena Hard Benchmark <arena_hard_meta_evaluation>`.


