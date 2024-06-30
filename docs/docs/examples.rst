.. _examples:
==============
Examples ✨
==============

Here you find complete examples showing how to perform different tasks using Unitxt. 
Each example is a self contained python file that you can run and later modify.


.. list-table:: 
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



.. list-table:: 
   :widths: 50 50 50 50
   :header-rows: 1

   * - What do you want to do?
     - Description
     - Link to code
     - Related documentation
   * - Evaluate your question-answering dataset  - using LLM as a judge
     - Demonstrates how to evaluate a user QA answering dataset in a standalone file using a user defined task and template. In addition, it shows how to define an LLM as a judge metric, specify the template it uses to produce the input to the judge, and select the judge model and platform.
     - `code <https://github.com/IBM/unitxt/blob/main/examples/standalone_evaluation_llm_as_judge.py>`_
     - | :ref:`LLM as a Judge Metrics Guide <llm_as_judge>`.
   * - Evaluate your summarization dataset  - using LLM as a judge
     - Demonstrates how to evaluate a document summarization dataset by define an LLM as a judge metric, specify the template it uses to produce the input to the judge, and select the judge model and platform.
     - `code <https://github.com/IBM/unitxt/blob/main/examples/evaluation_summarization_dataset_llm_as_judge>`_
     - | :ref:`LLM as a Judge Metrics Guide <llm_as_judge>`.


