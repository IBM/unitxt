.. _examples:
==============
Examples ✨
==============

Here you find complete examples of using Unitxt

.. list-table:: 
   :widths: 25 50 50 20
   :header-rows: 1

   * - Example
     - Description
     - Link to code
     - Related documentation
   * - Evaluate existing dataset 
     - Demonstrates how to evaluate an existing entailment dataset (wnli) using Huggingface 
       datasets and evaluate APIs, with no installation required.  
     - `evaluate_existing_dataset_no_install.py <https://github.com/IBM/unitxt/blob/main/examples/evaluate_existing_dataset_no_install.py>`_
     - :ref:`Evaluating datasets <evaluating_datasets>`.  
   * - Evaluate a user define open QA dataset 
     - Demonstrates how to evaluate a user QA answering dataset in a standalone file using a user defined task and template.
     - `standalone_qa_evaluation.py <https://github.com/IBM/unitxt/blob/main/examples/standalone_qa_evaluation.py>`_
     - :ref:`Add new dataset tutorial <adding_dataset>`.
   * - Evaluate a user defined open QA dataset while reusing existing catalog assets
     - Demonstrates how to evaluate a user QA answering dataset using the predefined open qa task and templates.
     - It also shows how to use preprocessing steps to  align the raw input of the dataset with the predefined task field.
     - `qa_evaluation.py <https://github.com/IBM/unitxt/blob/main/examples/qa_evaluation.py>`_
     - :ref:`Add new dataset tutorial <adding_dataset>`.
