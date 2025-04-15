.. _unitxt_evaluate_cli:

######################
Unitxt Evaluate CLI
######################

This document describes the command-line interface (CLI) provided by the ``evaluate_cli.py`` script for running language model evaluations using the ``unitxt`` library.

Overview
========

The ``unitxt-evaluate`` CLI streamlines the process of evaluating language models against diverse tasks defined within the ``unitxt`` framework. It manages:

* **Dataset Loading:** Loading and processing datasets according to specified ``unitxt`` recipes (cards, templates, formats, etc.).
* **Inference:** Generating model predictions using different backends:
    * Local Hugging Face models via ``transformers`` (``HFAutoModelInferenceEngine``).
    * Remote models accessed through APIs like OpenAI, Anthropic, Cohere, etc., often via ``litellm`` (``CrossProviderInferenceEngine``).
* **Evaluation:** Calculating metrics based on predictions and references.
* **Reporting:** Saving detailed results, configuration, and environment information to JSON files for analysis and reproducibility.

Usage
=====

The script is executed from the command line:

.. code-block:: bash

    python path/to/evaluate_cli.py --tasks <task_definitions> --model <model_type> --model_args <model_arguments> [options]

If ``evaluate_cli.py`` has been installed as an executable script (e.g., via ``pip install .`` with a ``pyproject.toml`` entry point), you might be able to run it directly:

.. code-block:: bash

    unitxt-evaluate --tasks <task_definitions> --model <model_type> --model_args <model_arguments> [options]

Example
-------

Evaluating a remote Llama 3 model on two variations of the BIRD text-to-SQL task, applying a chat template, limiting to 300 validation examples, and saving detailed samples:

.. code-block:: bash

    unitxt-evaluate \
        --tasks "card=cards.text2sql.bird,template=templates.text2sql.you_are_given_no_system+card=cards.text2sql.bird,template=templates.text2sql.you_are_given_no_system_with_hint" \
        --model cross_provider \
        --model_args "model_name=llama-3-1-405b-instruct,max_tokens=256" \
        --split validation \
        --limit 300 \
        --output_path ./results/bird_remote \
        --log_samples \
        --verbosity INFO \
        --trust_remote_code \
        --apply_chat_template \
        --batch_size 8

Arguments
=========

Task/Dataset Arguments
----------------------

.. option:: --tasks <task_definitions>, -t <task_definitions>

    **Required**. A plus-separated (``+``) list of one or more task definitions to evaluate. Each individual task definition is a comma-separated string of key-value pairs that specify the components of a ``unitxt`` recipe.

    * **Separator:** Use ``+`` to separate *different* task definitions if evaluating multiple variations or datasets in one run.
    * **Format (Single Task):** ``key1=value1,key2=value2,...``
    * **Format (Multiple Tasks):** ``key1=value1,key2=value2+keyA=valueA,keyB=valueB,...``
    * **Common Keys:** ``card``, ``template``, ``format``, ``num_demos``, ``max_train_instances``, ``max_validation_instances``, ``max_test_instances``, etc. Refer to ``unitxt`` documentation for available recipe parameters.
    * **Example (Single):** ``card=cards.mmlu,template=templates.mmlu.all,num_demos=5``
    * **Example (Multiple):** ``card=cards.mmlu,t=t.mmlu.all+card=cards.hellaswag,t=t.hellaswag.no`` (using shorthand ``t`` for ``template``)

.. option:: --split <split_name>

    The dataset split to load and evaluate (e.g., ``train``, ``validation``, ``test``). This should correspond to a split available in the specified card(s).
    * **Default:** ``test``

.. option:: --num_fewshots <N>

    Globally specifies the number of few-shot examples (demonstrations) to include in the prompt for *all* tasks defined in ``--tasks``.
    If set, this automatically adds/overrides the following parameters in each task's definition: ``num_demos=N``, ``demos_taken_from="train"``, ``demos_pool_size=-1``, ``demos_removed_from_data=True``.
    Using this will raise an error if ``num_demos`` is *also* specified directly within any task definition string in ``--tasks``, as it leads to ambiguity.
    * **Type:** integer
    * **Default:** ``None``

.. option:: --limit <N>, -L <N>

    Globally limits the number of examples loaded and evaluated *per task definition* for the specified ``--split``.
    This sets/overrides the ``max_<split_name>_instances`` parameter (e.g., ``max_test_instances`` if ``--split test``) for each task.
    Using this will raise an error if ``max_<split_name>_instances`` is *also* specified directly within any task definition string in ``--tasks``.
    * **Type:** integer
    * **Default:** ``None`` (evaluate all available examples in the split)

.. option:: --batch_size <N>, -b <N>

    The batch size for model inference. This parameter is primarily used by the local Hugging Face engine (``--model hf``) via ``HFAutoModelInferenceEngine``. Remote providers might handle batching differently or ignore this.
    * **Type:** integer
    * **Default:** ``1``

Model Arguments
---------------

.. option:: --model <model_type>, -m <model_type>

    Specifies the type of inference engine (and implicitly the model source) to use.
    * **Choices:** ``hf``, ``cross_provider``
    * **``hf``:** Use ``unitxt.inference.HFAutoModelInferenceEngine`` for models loadable via ``transformers.AutoModel``. Typically used for local models or those on the Hugging Face Hub. Requires ``pretrained=<model_id_or_path>`` in ``--model_args``.
    * **``cross_provider``:** Use ``unitxt.inference.CrossProviderInferenceEngine``, which often leverages ``litellm`` to interact with various model APIs (OpenAI, Anthropic, Cohere, Vertex AI, self-hosted endpoints, etc.). Requires ``model_name=<provider/model_id>`` (e.g., ``openai/gpt-4o``, ``anthropic/claude-3-opus-20240229``) in ``--model_args``.
    * **Default:** ``hf``

.. option:: --model_args <arguments>, -a <arguments>

    Arguments passed directly to the constructor of the selected inference engine (``HFAutoModelInferenceEngine`` or ``CrossProviderInferenceEngine``), *after* required keys (``pretrained`` or ``model_name``) are extracted. Can be provided as a comma-separated string of key-value pairs or as a JSON string.
    * **Format (Key-Value):** ``key1=value1,key2=value2,...`` (Values automatically typed as int, float, bool, or string). Example: ``torch_dtype=bfloat16,device=cuda,trust_remote_code=true``
    * **Format (JSON):** ``'{"key1": "value1", "key2": 123, "key3": true}'`` (Use double quotes for JSON keys and string values).
    * **Required Keys:**
        * For ``--model hf``: Must include ``pretrained=<model_id_or_path>``.
        * For ``--model cross_provider``: Must include ``model_name=<provider/model_id>``.
    * **Engine-Specific Args:** Refer to the documentation for ``HFAutoModelInferenceEngine`` and ``CrossProviderInferenceEngine`` (and potentially ``litellm`` for ``cross_provider``) for available arguments (e.g., ``torch_dtype``, ``device``, ``quantization_config`` for ``hf``; ``api_base``, ``api_key``, ``max_tokens``, ``temperature`` for ``cross_provider``). Note: Sensitive keys like ``api_key`` are often better handled via environment variables.
    * **Merging with ``--gen_kwargs``:** Arguments from ``--gen_kwargs`` are merged into this dictionary *before* initializing the inference engine. See ``--gen_kwargs`` description.
    * **Default:** ``{}``

.. option:: --gen_kwargs <arguments>

    Additional key-value arguments intended specifically for the model's generation process (e.g., parameters for ``model.generate()`` in Transformers or equivalent API call parameters). Format is the same as ``--model_args`` (key-value string or JSON).
    These arguments are **merged** into the arguments from ``--model_args`` before the inference engine is initialized. If a key exists in both ``--model_args`` and ``--gen_kwargs``, the value from ``--gen_kwargs`` will take precedence.
    * **Example:** ``temperature=0,top_p=0.9,max_new_tokens=100``
    * **Default:** ``None``

.. option:: --chat_template_kwargs <arguments>

    Key-value arguments passed directly to the tokenizer's ``apply_chat_template`` method. This is only relevant if ``--apply_chat_template`` is also used. Format is the same as ``--model_args`` (key-value string or JSON). Refer to the `Hugging Face Transformers documentation <https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template>`_ for available arguments.
    * **Example:** ``thinking=True,add_generation_prompt=True``
    * **Default:** ``None``

.. option:: --apply_chat_template

    If specified, the script will automatically set the task format to ``formats.chat_api`` for all tasks defined in ``--tasks``. This format uses the tokenizer's ``apply_chat_template`` method to structure the input.
    Using this flag will raise an error if ``format`` is *also* specified directly within any task definition string in ``--tasks``.
    * **Default:** ``False`` (uses the format specified in the task definition or ``unitxt`` defaults).

Output and Logging Arguments
----------------------------

.. option:: --output_path <path>, -o <path>

    Directory where the output JSON files will be saved. The directory will be created if it doesn't exist.
    * **Default:** ``.`` (current directory)

.. option:: --output_file_prefix <prefix>

    A prefix used for naming the output JSON files. A timestamp (``YYYY-MM-DDTHH:MM:SS``) is automatically prepended to ensure unique filenames.
    * **Example:** If ``--output_file_prefix results_run1``, files might be named ``2025-04-14T10:05:14_results_run1.json`` and ``2025-04-14T10:05:14_results_run1_samples.json``.
    * **Default:** ``evaluation_results``

.. option:: --log_samples, -s

    If specified, a detailed file containing data for each individual evaluated instance will be saved alongside the summary results file.
    * **Filename:** ``<timestamp>_<prefix>_samples.json``
    * **Default:** ``False`` (only the summary results file is saved).

.. option:: --verbosity <level>, -v <level>

    Controls the level of detail in log messages printed to the console.
    * **Choices:** ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL`` (case-insensitive)
    * **Default:** ``INFO``

Unitxt Settings
---------------

These arguments configure underlying ``unitxt`` or Hugging Face ``datasets`` behavior.

.. option:: --trust_remote_code

    Allows the execution of Python code defined in remote Hugging Face Hub repositories (e.g., custom code within dataset loading scripts or metrics). **Warning:** Only enable this if you trust the source of the code.
    * **Default:** ``False``

.. option:: --disable_hf_cache

    Disables the caching mechanism used by the Hugging Face ``datasets`` library. This forces datasets to be redownloaded and reprocessed.
    * **Default:** ``False``

.. option:: --cache_dir <path>

    Specifies a custom directory for the Hugging Face ``datasets`` cache. This overrides the default location (usually ``~/.cache/huggingface/datasets``) and the ``HF_DATASETS_CACHE`` / ``HF_HOME`` environment variables for operations within this script.
    * **Default:** ``None`` (uses default cache location or environment variables).

Output Files
============

The CLI generates one or two JSON files in the specified ``--output_path``.

1.  **Results Summary File** (``<timestamp>_<prefix>.json``)
    Contains aggregated scores and execution environment details.

    * **``environment_info``** (object): Details about the execution context:
        * ``timestamp_utc`` (string): Timestamp of evaluation completion (UTC, ISO 8601).
        * ``command_line_invocation`` (list): The command-line arguments used (``sys.argv``).
        * ``parsed_arguments`` (object): Dictionary representation of the parsed command-line arguments.
        * ``unitxt_version`` (string): Installed ``unitxt`` package version (or "N/A").
        * ``unitxt_commit_hash`` (string): Git commit hash of ``unitxt`` installation (or "N/A").
        * ``python_version`` (string): Python interpreter version.
        * ``system`` (string): OS name (e.g., "Linux", "Darwin", "Windows").
        * ``system_version`` (string): OS version details.
        * ``installed_packages`` (object): Dictionary mapping installed Python packages to their versions.
    * **``results``** (object): Contains the evaluation scores.
        * Keys are the task definition strings exactly as provided in the ``--tasks`` argument.
        * Values are objects containing the calculated metrics for that specific task (e.g., ``"accuracy": 0.85``, ``"score": 0.85``, ``"score_name": "accuracy"``, potentially confidence intervals like ``"accuracy_ci_low"``, ``"accuracy_ci_high"``).
        * May also include overall summary metrics across all tasks evaluated (e.g., a top-level ``"score"`` key representing the mean score, often accompanied by ``"score_name": "subsets_mean"``).

2.  **Detailed Samples File** (``<timestamp>_<prefix>_samples.json``)
    Generated only if ``--log_samples`` is specified. Contains instance-level details.

    * **``environment_info``** (object): Same structure as in the summary file.
    * **``samples``** (object): A dictionary where keys are the task definition strings from ``--tasks``.
        * Each value is a list of objects, where each object represents one evaluated instance.
        * Instance object keys typically include:
            * ``source``: Original input data record.
            * ``processed``: Input potentially transformed by the recipe (e.g., formatted prompt). May not always be present.
            * ``prediction``: Raw output generated by the model.
            * ``references``: List of ground truth reference(s).
            * ``metrics``: Dictionary of scores calculated for this specific instance.
            * ``task_data``: Additional metadata from the ``unitxt`` processing steps.
            * *Note:* The ``postprocessors`` key used during internal computation is removed before saving.

Frequently Asked Questions (FAQ)
================================

**Q: Why does ``--tasks`` use ``+`` as a separator? Why not commas or semicolons?**
A: The ``+`` separates distinct task *definitions*. Since each task definition *itself* is a comma-separated list of key-value pairs (e.g., ``card=c,template=t``), using commas or semicolons to separate multiple tasks would be ambiguous. The ``+`` provides a clear delimiter between full task recipes.

**Q: What's the difference between ``--model_args`` and ``--gen_kwargs``?**
A: Both allow passing key-value arguments.
* ``--model_args`` are primarily intended for arguments needed to *initialize* the inference engine (e.g., ``pretrained``, ``device``, ``torch_dtype``, ``model_name``, ``max_tokens``).
* ``--gen_kwargs`` are intended for arguments controlling the *generation process* itself (e.g., ``temperature``, ``top_p``, ``do_sample``).
* **Important:** Arguments from ``--gen_kwargs`` are merged into ``--model_args`` *before* the engine is initialized, with ``--gen_kwargs`` values overwriting any conflicting keys from ``--model_args``.

**Q: I'm getting `AttributeError: 'Namespace' object has no attribute 'batch_size'` (or similar) in my tests.**
A: When manually creating an `argparse.Namespace` object in a test (e.g., `args = argparse.Namespace(...)`), ensure you include *all* attributes that the code under test might access, even if they have default values in the real parser. Check the `setup_parser` function for defaults (like `batch_size=1`).

**Q: I'm getting `UnitxtArtifactNotFoundError: Artifact 'some_name' does not exist...`**
A: This means ``unitxt`` cannot find an artifact (like a card, template, metric) you specified.
* Double-check the spelling and full name (e.g., ``cards.common_sense.hellaswag``) in your ``--tasks`` definition.
* Ensure the artifact exists in the default ``unitxt`` catalog or any custom catalog paths you might have configured.
* Check for typos in keys (e.g., `templete=` instead of `template=`).

**Q: The CLI fails with an error about invalid JSON for ``--model_args`` (or ``--gen_kwargs`` / ``--chat_template_kwargs``).**
A: If providing arguments as a JSON string, ensure it's valid:
* Wrap the entire JSON string in single quotes (for the shell) or escape double quotes appropriately.
* Use double quotes (``"``) for all keys and string values *inside* the JSON.
* Example: ``--model_args '{"pretrained": "my/model", "some_flag": true, "count": 10}'``

**Q: I get `ValueError: Argument 'pretrained' is required...` or `ValueError: Argument 'model_name' is required...`**
A: You must provide the correct identifier key within ``--model_args`` based on your selected ``--model`` type:
* If ``--model hf``, include ``pretrained=<model_id_or_path>`` in ``--model_args``.
* If ``--model cross_provider``, include ``model_name=<provider/model_id>`` in ``--model_args``.

**Q: How do global arguments like ``--limit``, ``--num_fewshots``, ``--apply_chat_template`` interact with task-specific arguments in ``--tasks``?**
A: The global CLI arguments generally take precedence.
* If you provide ``--limit N``, it will set `max_<split>_instances=N` for all tasks, potentially overwriting values set inside the ``--tasks`` string. The script includes checks to error out if you provide *both* the CLI argument and a corresponding key within the *same* task string in ``--tasks`` (e.g., ``--limit 10`` and ``...,max_test_instances=5`` in ``--tasks`` when ``--split test``).
* Similar precedence and conflict checks apply to ``--num_fewshots`` (vs ``num_demos``) and ``--apply_chat_template`` (vs ``format``).

**Q: Where do I put API keys (like OpenAI API key) for ``--model cross_provider``?**
A: For security, **do not** pass sensitive API keys directly via ``--model_args``. ``CrossProviderInferenceEngine`` typically relies on ``litellm``, which finds keys through standard methods:
* **Environment Variables:** (Recommended) Set environment variables like ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, etc., before running the script.
* **LiteLLM Config File:** Configure keys in a `litellm` configuration file.
* Refer to the `litellm` documentation for managing API keys.

**Q: The `unitxt_commit_hash` in my output is "N/A". Why?**
A: The script tries to get the commit hash using the `git rev-parse HEAD` command within the detected installation directory of the `unitxt` package. This might fail if:
* The `unitxt` package was not installed from a Git repository (e.g., installed from PyPI as a standard package).
* The `git` command is not available in your system's PATH.
* The script cannot correctly determine the `unitxt` package location or it's not within a recognizable Git repository structure.

Troubleshooting
===============

* **Argument Parsing Errors:** Double-check formatting for JSON/key-value strings, ensure required keys like ``pretrained``/``model_name`` are present, and verify the ``+`` separator for ``--tasks``.
* **Artifact Not Found Errors:** Verify artifact names (cards, templates, etc.) and catalog accessibility. Check for typos.
* **Dependency Errors:** Ensure ``unitxt``, ``datasets``, ``transformers`` are installed. For ``hf`` models, ``torch`` and possibly ``accelerate`` are needed. For ``cross_provider``, ``litellm`` and potentially provider-specific libraries (like ``openai``) are needed.
* **Remote Model Errors (cross_provider):** Verify API keys (via environment variables), model identifiers (e.g., ``openai/gpt-4o``), quotas, network connectivity, and any necessary ``litellm`` configuration.
* **CUDA/Device Errors (hf):** Ensure GPU drivers/CUDA toolkit are correctly installed and configured if using ``device=cuda`` in ``--model_args``. Check available GPU memory.
* **Conflicting Arguments:** Avoid specifying arguments both globally (e.g., ``--limit``) and within the ``--tasks`` string for the same parameter (e.g., ``max_test_instances``) – the script should raise an error if this happens.