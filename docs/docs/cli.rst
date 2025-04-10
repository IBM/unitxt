.. _unitxt_evaluate_cli:

######################
Unitxt Evaluate CLI
######################

This document describes the command-line interface (CLI) for running evaluations using the ``unitxt`` library via the ``evaluate_cli.py`` script.

Overview
========

The ``unitxt-evaluate`` CLI provides a convenient way to evaluate language models on various tasks defined within the ``unitxt`` framework. It handles dataset loading, model inference (supporting both local Hugging Face models and remote models via ``CrossProviderInferenceEngine``), evaluation execution, and results reporting.

Usage
=====

The script is typically run from the command line. Here's the basic structure:

.. code-block:: bash

   python path/to/evaluate_cli.py --tasks <task_definitions> --model <model_type> --model_args <model_arguments> [options]

Example
-------

Here's an example command evaluating a Llama 3 model hosted remotely on two different BIRD task variations:

.. code-block:: bash

   unitxt-evaluate \\
       --tasks "card=cards.text2sql.bird,template=templates.text2sql.you_are_given_no_system+card=cards.text2sql.bird,template=templates.text2sql.you_are_given_no_system_with_hint" \\
       --model cross_provider \\
       --model_args "model_name=llama-3-1-405b-instruct,max_tokens=256" \\
       --split validation \\
       --limit 300 \\
       --output_path ./results/bird_remote \\
       --log_samples \\
       --verbosity INFO \\
       --trust_remote_code \\
       --apply_chat_template \\
       --batch_size 8

**Note:** Replace ``unitxt-evaluate`` with ``python path/to/evaluate_cli.py`` if the script is not installed as an executable in your environment's PATH.

Arguments
=========

Task/Dataset Arguments
----------------------

.. option:: --tasks <task_definitions>, -t <task_definitions>

   **Required**. A plus-separated (``+``) list of task definitions. Each task definition is a comma-separated string of key-value pairs specifying the components of the task (e.g., card, template, format, etc.).

   * **Format:** ``key1=value1,key2=value2+key3=value3,...``
   * **Example:** ``card=cards.mmlu,template=templates.mmlu.all+card=cards.hellaswag,template=templates.hellaswag.no_choice``

.. option:: --split <split_name>

   The dataset split to use (e.g., ``train``, ``validation``, ``test``).
   * **Default:** ``test``

.. option:: --num_fewshots <N>

   Specifies the number of few-shot examples to include in the prompt. If set, it automatically configures ``num_demos``, ``demos_taken_from``, ``demos_pool_size``, and ``demos_removed_from_data`` in the task arguments. Cannot be used if ``num_demos`` is already specified within a task definition in ``--tasks``.
   * **Default:** ``None``

.. option:: --limit <N>, -L <N>

   Limits the number of examples evaluated per task definition. This sets the ``max_<split_name>_instances`` parameter for each task. Cannot be used if ``max_<split_name>_instances`` is already specified within a task definition in ``--tasks``.
   * **Default:** ``None`` (evaluate all examples in the specified split)

.. option:: --batch_size <N>, -b <N>

    The batch size to use during inference, specifically when using the ``hf`` model type (``HFAutoModelInferenceEngine``).
    * **Default:** ``1``

Model Arguments
---------------

.. option:: --model <model_type>, -m <model_type>

   Specifies the type of inference engine to use.
   * **Choices:** ``hf``, ``cross_provider``
   * **``hf``:** Uses ``HFAutoModelInferenceEngine`` for local Hugging Face models. Requires ``pretrained=<model_id_or_path>`` in ``--model_args``.
   * **``cross_provider``:** Uses ``CrossProviderInferenceEngine`` for remote models (e.g., via APIs like LiteLLM). Requires ``model_name=<provider/model_id>`` in ``--model_args``.
   * **Default:** ``hf``

.. option:: --model_args <arguments>, -a <arguments>

   Arguments passed to the selected inference engine. Can be provided as a comma-separated string of key-value pairs or as a JSON string.
   * **Format (Key-Value):** ``key1=value1,key2=value2,...`` (Values are automatically typed as int, float, bool, or string)
   * **Format (JSON):** ``'{"key1": "value1", "key2": 123}'`` (Use double quotes for JSON keys and string values)
   * **Required Keys:**
        * For ``--model hf``: ``pretrained=<model_id_or_path>``
        * For ``--model cross_provider``: ``model_name=<provider/model_id>``
   * **Examples:**
        * ``hf``: ``pretrained=meta-llama/Llama-3.1-8B-Instruct,torch_dtype=bfloat16,device=cuda``
        * ``cross_provider``: ``model_name=openai/gpt-4o,max_tokens=512,temperature=0.5``
   * **Default:** ``{}``

.. option:: --gen_kwargs <arguments>

   Additional generation arguments specifically for the model, especially relevant for tasks requiring greedy generation (e.g., ``greedy_until`` postprocessor). Passed to the model's generation function. Format is the same as ``--model_args`` (key-value string or JSON).
   * **Example:** ``temperature=0,top_p=0.9``
   * **Default:** ``None``

.. option:: --chat_template_kwargs <arguments>

   Arguments passed to the tokenizer's ``apply_chat_template`` method when ``--apply_chat_template`` is used. Format is the same as ``--model_args`` (key-value string or JSON).
   * **Example:** ``thinking=True`` (Refer to Hugging Face Transformers documentation for available tokenizer arguments)
   * **Default:** ``None``

.. option:: --apply_chat_template

   If set, applies the model's chat template (via ``tokenizer.apply_chat_template``) to the input. This automatically sets the task format to ``formats.chat_api``. Cannot be used if ``format`` is already specified within a task definition in ``--tasks``.
   * **Default:** ``False``

Output and Logging Arguments
----------------------------

.. option:: --output_path <path>, -o <path>

   Directory where the evaluation results and logs will be saved.
   * **Default:** ``.`` (current directory)

.. option:: --output_file_prefix <prefix>

   Prefix for the output JSON file names. The final filenames will be ``<timestamp>_<prefix>.json`` and ``<timestamp>_<prefix>_samples.json``.
   * **Default:** ``evaluation_results``

.. option:: --log_samples, -s

   If set, saves detailed information for each evaluated instance (including source, preprocessed input, prediction, references, scores) to a separate ``<timestamp>_<prefix>_samples.json`` file.
   * **Default:** ``False``

.. option:: --verbosity <level>, -v <level>

   Controls the logging level.
   * **Choices:** ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``
   * **Default:** ``INFO``

Unitxt Settings
---------------

.. option:: --trust_remote_code

   Allows the execution of potentially unverified code from the Hugging Face Hub, which might be required by certain datasets or metrics within ``unitxt``. Use with caution.
   * **Default:** ``False``

.. option:: --disable_hf_cache

   Disables the Hugging Face ``datasets`` library caching mechanism.
   * **Default:** ``False``

.. option:: --cache_dir <path>

   Specifies a custom directory for the Hugging Face ``datasets`` cache, overriding the default location or the ``HF_DATASETS_CACHE`` environment variable.
   * **Default:** ``None``

Output Files
============

The CLI generates one or two JSON files in the specified ``--output_path``, prefixed with a timestamp and the ``--output_file_prefix``.

1.  **Results Summary File** (``<timestamp>_<prefix>.json``)
    This file contains the overall evaluation results and environment information.

    * ``environment_info``: A dictionary containing details about the execution environment:
        * ``timestamp_utc``: Time the evaluation finished (ISO format).
        * ``command_line_invocation``: The exact command used to run the script.
        * ``parsed_arguments``: The arguments as parsed by the script.
        * ``unitxt_version``: Installed ``unitxt`` package version.
        * ``unitxt_commit_hash``: Git commit hash of the installed ``unitxt`` (if available).
        * ``python_version``: Python interpreter version.
        * ``system``: Operating system name (e.g., "Linux", "Darwin").
        * ``system_version``: OS version details.
        * ``installed_packages``: A dictionary of installed Python packages and their versions.
    * ``results``: A dictionary containing the aggregated scores for each task definition evaluated, plus overall scores (like ``score`` which is the mean score across subsets).
        * Each key corresponds to a task definition string from the ``--tasks`` argument.
        * The value is a dictionary of metrics (e.g., ``accuracy``, ``f1``, ``rougeL``, ``score``, ``score_name``, confidence intervals if available, etc.) calculated for that task.
        * An overall ``score`` and ``score_name`` (e.g., ``subsets_mean``) summarizing the performance across all tasks might also be present.

    *(See the example JSON provided in the prompt for a detailed structure)*

2.  **Detailed Samples File** (``<timestamp>_<prefix>_samples.json``)
    This file is generated only if ``--log_samples`` is specified. It contains detailed information for every instance processed during the evaluation.

    * ``environment_info``: Same as in the results summary file.
    * ``samples``: A dictionary where keys are the task definition strings from ``--tasks``. The values are lists, where each element in the list is a dictionary representing one evaluated instance. This instance dictionary typically includes:
        * ``source``: The original input data for the instance.
        * ``processed``: The input after applying the ``unitxt`` recipe (formatting, few-shot examples, etc.).
        * ``prediction``: The raw output from the model.
        * ``references``: The ground truth or target output(s).
        * ``metrics``: Scores calculated specifically for this instance.
        * ``task_data``: Additional metadata related to the task processing.

Troubleshooting
===============

* **Argument Parsing Errors:**
    * Ensure JSON strings in ``--model_args``, ``--gen_kwargs``, or ``--chat_template_kwargs`` use double quotes for keys and string values (e.g., ``'{"key": "value"}'``).
    * For key-value string format, ensure keys and values are separated by ``=`` and pairs by ``,``.
    * Check that required arguments (like ``pretrained`` for ``hf`` or ``model_name`` for ``cross_provider`` in ``--model_args``) are provided.
* **Artifact Not Found Errors:** Verify that the card, template, or other artifact names used in ``--tasks`` are correct and accessible in the ``unitxt`` catalog or specified paths.
* **Dependency Errors:** Ensure all necessary libraries (``unitxt``, ``datasets``, ``transformers``, potentially ``torch``, ``accelerate``, ``openai``, ``litellm`` depending on the model type) are installed.
* **Remote Model Errors:** Check API keys, model availability, and rate limits if using ``--model cross_provider``. Ensure ``litellm`` is configured correctly if needed.
* **CUDA/Device Errors:** If using ``--model hf`` with a GPU, ensure CUDA is set up correctly and the specified ``device`` in ``--model_args`` is available.
