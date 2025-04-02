.. _evaluate_cli:

================
evaluate_cli.py
================

Command-Line Interface for Unitxt Evaluation
--------------------------------------------

This script provides a command-line interface (CLI) to run evaluations using the ``unitxt`` library. It allows users to specify a dataset (card and template), a model (local Hugging Face or remote via CrossProvider), evaluation parameters, and output options directly from the terminal.

Core Functionality
------------------

1.  **Load Data**: Loads a dataset based on a ``unitxt`` task string (card, template, etc.) and split.
2.  **Initialize Model**: Sets up the inference engine, either a local Hugging Face model (`HFAutoModelInferenceEngine`) or a remote model (`CrossProviderInferenceEngine`).
3.  **Run Inference**: Generates predictions for the loaded dataset using the specified model.
4.  **Run Evaluation**: Calculates evaluation metrics by comparing the predictions against the dataset references.
5.  **Save Results**: Outputs the global evaluation scores and, optionally, detailed instance-level results (source, prediction, references, metrics) to JSON files.

Usage
-----

The script is typically run using an entry point like `unitxt-eval` (assuming it's set up in `pyproject.toml` or similar).

**Basic Syntax:**

.. code-block:: bash

   unitxt-eval --tasks <task_string> [options]

**Examples:**

* **Evaluating with a remote model (CrossProvider):**

    .. code-block:: bash

       unitxt-eval \
           --tasks "card=cards.text2sql.bird,template=templates.text2sql.you_are_given_with_hint_with_sql_prefix" \
           --split validation \
           --model cross_provider \
           --model_args "model_name=llama-3-3-70b-instruct,max_tokens=256" \
           --limit 100 \
           --output_dir ./debug_output/bird_remote \
           --log_samples

* **Evaluating with a local Hugging Face model:**

    .. code-block:: bash

       unitxt-eval \
           --tasks "card=cards.text2sql.bird,template=templates.text2sql.you_are_given_with_hint_with_sql_prefix" \
           --split validation \
           --model hf \
           --model_args "pretrained=HuggingFaceTB/SmolLM2-135M-Instruct,max_new_tokens=256" \
           --limit 100 \
           --output_dir ./debug_output/bird_local \
           --log_samples

Command-Line Arguments
----------------------

.. option:: --tasks <task_string>, -t <task_string>

   **Required**. Unitxt task/dataset identifier string.
   Format: ``'card=<card_ref>,template=<template_ref>,...'``
   Example: ``'card=cards.mmlu,template=templates.mmlu.all_5_shot'``

.. option:: --split <split_name>

   Dataset split to use (e.g., 'train', 'validation', 'test').
   Default: ``test``.

.. option:: --limit <N>, -L <N>

   Limit the number of examples per task/dataset. If not specified, all examples in the split are used.

.. option:: --model <model_type>

   Specifies the model type/engine.
   Choices: ``hf``, ``cross_provider``.
   Default: ``hf``.
   - ``hf``: Local Hugging Face model via `HFAutoModelInferenceEngine`. Requires ``pretrained=<model_id_or_path>`` in ``--model_args``.
   - ``cross_provider``: Remote model via `CrossProviderInferenceEngine`. Requires ``model_name=<provider/model_id>`` in ``--model_args``.

.. option:: --model_args <args_string_or_json>, -a <args_string_or_json>

   Comma-separated string (``key=value,key2=value2``) or JSON formatted arguments for the model/inference engine.
   - For ``--model hf``: ``pretrained`` key is **required**. Other args (e.g., ``torch_dtype``, ``device``, generation params) are passed to the model/tokenizer/generation.
     Example: ``'pretrained=meta-llama/Llama-3.1-8B-Instruct,torch_dtype=bfloat16,device=cuda'``
   - For ``--model cross_provider``: ``model_name`` key is **required**. Other args (e.g., ``max_tokens``, ``temperature``) are passed to the inference engine.
     Example: ``'model_name=openai/gpt-4o,max_tokens=512,temperature=0.7'``
   - JSON Example: ``'{"pretrained": "my_model", "torch_dtype": "float32"}'``

.. option:: --output_dir <path>, -o <path>

   Directory to save evaluation results and logs.
   Default: ``.`` (current directory).

.. option:: --output_file_prefix <prefix>

   Prefix for the output JSON file names (``<prefix>.json`` and ``<prefix>_samples.json``).
   Default: ``evaluation_results``.

.. option:: --log_samples, -s

   If specified, save individual predictions and scores to a separate ``<prefix>_samples.json`` file.

.. option:: --verbosity <level>, -v <level>

   Controls logging verbosity level.
   Choices: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``.
   Default: ``INFO``.

.. option:: --trust_remote_code

   Allow execution of unverified code from the HuggingFace Hub (used by ``datasets``/``unitxt``). Use with caution.
   Default: ``False``.

.. option:: --disable_hf_cache

   Disable HuggingFace ``datasets`` caching.
   Default: ``False``.

.. option:: --cache_dir <path>

   Directory for HuggingFace ``datasets`` cache (overrides default and ``HF_DATASETS_CACHE`` environment variable if set).

Key Functions
-------------

* ``main()``: Parses arguments, sets up logging and settings, orchestrates the loading, inference, evaluation, and saving process. Handles exceptions.
* ``setup_parser()``: Defines the ``argparse`` configuration for all CLI arguments.
* ``try_parse_json()`` / ``_parse_key_value_string()``: Parses the ``--model_args`` input, handling both JSON and key=value string formats.
* ``configure_unitxt_settings()``: Applies ``unitxt`` global settings based on arguments (caching, remote code).
* ``load_data()``: Loads the specified dataset using ``unitxt.load_dataset``.
* ``prepare_model_args()``: Processes the parsed model arguments.
* ``initialize_inference_engine()``: Creates the appropriate ``InferenceEngine`` instance (``HFAutoModelInferenceEngine`` or ``CrossProviderInferenceEngine``).
* ``run_inference()``: Calls the ``infer()`` method of the engine.
* ``run_evaluation()``: Calls the ``unitxt.evaluate`` function.
* ``process_and_save_results()`` / ``_extract_scores_and_samples()`` / ``_save_results_to_disk()``: Extracts scores, formats results, prints a summary to the console, and saves results to JSON files.

Output Files
------------

1.  **`<output_dir>/<prefix>.json`**: Contains the configuration used for the run and the aggregated global scores.

    * **Example Structure:**

      .. code-block:: json

         {
             "config": {
                 "tasks": "card=cards.text2sql.bird,template=templates.text2sql.you_are_given_with_hint_with_sql_prefix",
                 "split": "validation",
                 "limit": 100,
                 "model": "generic_remote",
                 "model_args": {
                     "max_tokens": 256
                 },
                 "output_dir": "./debug_output/bird_remote",
                 "output_file_prefix": "evaluation_results",
                 "log_samples": true,
                 "verbosity": "INFO",
                 "trust_remote_code": true,
                 "disable_hf_cache": false,
                 "cache_dir": null
             },
             "global_scores": {
                 "num_of_instances": 100,
                 "anls": 0.5449067219688716,
                 "score": 0.45,
                 "score_name": "non_empty_execution_accuracy",
                 "sqlparse_equivalence": 0.03,
                 "sql_exact_match": 0.05,
                 "sqlglot_validity": 1.0,
                 "sqlparse_validity": 1.0,
                 "sqlglot_optimized_equivalence": 0.08,
                 "sqlglot_equivalence": 0.06,
                 "sqlparse_equivalence_ci_low": 0.01,
                 "sqlparse_equivalence_ci_high": 0.08,
                 "sql_exact_match_ci_low": 0.02,
                 "sql_exact_match_ci_high": 0.1,
                 "sqlglot_optimized_equivalence_ci_low": 0.04,
                 "sqlglot_optimized_equivalence_ci_high": 0.15,
                 "sqlglot_equivalence_ci_low": 0.02,
                 "sqlglot_equivalence_ci_high": 0.12,
                 "score_ci_low": 0.35,
                 "score_ci_high": 0.55,
                 "gold_error": 0.0,
                 "predicted_error": 0.06,
                 "non_empty_execution_accuracy": 0.45,
                 "execution_accuracy": 0.45,
                 "predicted_sql_runtime": 0.0019813596113817766,
                 "gold_sql_runtime": 0.0024379391391994433,
                 "pred_to_gold_runtime_ratio": 1.1805156552830012,
                 "subset_non_empty_execution_result": 0.52,
                 "non_empty_gold_df": 0.94,
                 "non_empty_execution_accuracy_ci_low": 0.35,
                 "non_empty_execution_accuracy_ci_high": 0.55,
                 "execution_accuracy_ci_low": 0.35,
                 "execution_accuracy_ci_high": 0.55,
                 "predicted_sql_runtime_ci_low": 0.0017113976708526206,
                 "predicted_sql_runtime_ci_high": 0.0022960377183659773,
                 "gold_sql_runtime_ci_low": 0.002035246512971937,
                 "gold_sql_runtime_ci_high": 0.003492967311873054,
                 "subset_non_empty_execution_result_ci_low": 0.42,
                 "subset_non_empty_execution_result_ci_high": 0.62
             }
         }

    * **Parsing Example (Python):**

      To extract the main score and its confidence interval:

      .. code-block:: python

         import json

         # Assume 'results.json' is the path to your output file
         file_path = 'results.json' # Or './debug_output/bird_remote/evaluation_results.json' etc.

         try:
             with open(file_path, 'r', encoding='utf-8') as f:
                 data = json.load(f)

             # Access the global scores dictionary
             global_scores = data.get('global_scores', {})

             # Extract specific fields (use .get() for safety)
             score = global_scores.get('score')
             score_name = global_scores.get('score_name')
             score_ci_low = global_scores.get('score_ci_low')
             score_ci_high = global_scores.get('score_ci_high')

             if score is not None and score_name is not None:
                 print(f"Score Name: {score_name}")
                 print(f"Score: {score}")
                 if score_ci_low is not None and score_ci_high is not None:
                     print(f"Confidence Interval: [{score_ci_low}, {score_ci_high}]")
                 else:
                     print("Confidence interval not found.")
             else:
                 print("Could not find 'score' or 'score_name' in global_scores.")

         except FileNotFoundError:
             print(f"Error: File not found at {file_path}")
         except json.JSONDecodeError:
             print(f"Error: Could not decode JSON from {file_path}")
         except Exception as e:
             print(f"An unexpected error occurred: {e}")


2.  **`<output_dir>/<prefix>_samples.json`** (only if ``--log_samples`` is used): Contains a list of dictionaries, one for each instance, including source text, prediction, references, instance-level metrics, and original task data.

