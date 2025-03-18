.. _settings:

=====================================
General Settings
=====================================

.. _settings_intro:

Introduction
============
Unitxt allows you to control various features at the library level, such as logging, caching, and more. These settings enable you to customize the behavior of the library to better suit your needs.

Settings in Unitxt can be modified using Python:

.. code-block:: python

    import unitxt

    unitxt.settings.default_verbosity = "info"

Or by setting environment variables:

.. code-block:: bash

    export UNITXT_DEFAULT_VERBOSITY="debug"

The naming convention for environment variables follows the pattern of prefixing the setting name with ``UNITXT_`` in uppercase.


Important Settings
==================

Logging Level
-------------
- ``unitxt.settings.default_verbosity``: Defines the verbosity level of the Python logging logger.
  - Type: str | Default: "info" | Env Var: ``UNITXT_DEFAULT_VERBOSITY``

Data Loaders
------------
- ``unitxt.settings.global_loader_limit``: Limits instances loaded by any data loader.
  - Type: int | Default: None | Env Var: ``UNITXT_GLOBAL_LOADER_LIMIT``
- ``unitxt.settings.loaders_max_retries``: Maximum number of retries for loading data.
  - Type: int | Default: 10 | Env Var: ``UNITXT_LOADERS_MAX_RETRIES``
- ``unitxt.settings.loader_cache_size``: Defines the in-memory cache size for loaders.
  - Type: int | Default: 25 | Env Var: ``UNITXT_LOADER_CACHE_SIZE``

External Code
-------------
- ``unitxt.settings.allow_unverified_code``: Allows execution of unverified external code.
  - Type: bool | Default: False | Env Var: ``UNITXT_ALLOW_UNVERIFIED_CODE``

Hugging Face Integration
------------------------
- ``unitxt.settings.disable_hf_datasets_cache``: Disables caching for Hugging Face datasets.
  - Type: bool | Default: False | Env Var: ``UNITXT_DISABLE_HF_DATASETS_CACHE``
- ``unitxt.settings.stream_hf_datasets_by_default``: Enables streaming mode for Hugging Face datasets by default.
  - Type: bool | Default: False | Env Var: ``UNITXT_STREAM_HF_DATASETS_BY_DEFAULT``
- ``unitxt.settings.hf_offline_models_path``: Specifies the path to the directory containing offline pre-downloaded Hugging Face models. You will need to manually download the assets to the directory.
  - Type: str | Default: None | Env Var: ``UNITXT_HF_OFFLINE_MODELS_PATH``
- ``unitxt.settings.hf_offline_metrics_path``: Specifies the path to the directory containing offline pre-downloaded Hugging Face metrics. You will need to manually download the assets to the directory.
  - Type: str | Default: None | Env Var: ``UNITXT_HF_OFFLINE_METRICS_PATH``
- ``unitxt.settings.hf_offline_datasets_path``: Specifies the path to the directory containing offline pre-downloaded Hugging Face datasets. You will need to manually download the assets to the directory.
  - Type: str | Default: None | Env Var: ``UNITXT_HF_OFFLINE_DATASETS_PATH``


Randomness
----------
- ``unitxt.settings.seed``: Seed value for ensuring reproducibility.
  - Type: int | Default: 42 | Env Var: ``UNITXT_SEED``

LLM Inference
-------------
- ``unitxt.settings.mock_inference_mode``: Enables or disables mock inference mode.
  - Type: bool | Default: False | Env Var: ``UNITXT_MOCK_INFERENCE_MODE``
- ``unitxt.settings.default_provider``: Specifies the default inference provider for ``CrossProviderInferenceEngine``s.
  - Type: str | Default: "watsonx" | Env Var: ``UNITXT_DEFAULT_PROVIDER``

Catalogs
--------
- ``unitxt.settings.catalogs``: Defines list of directories with local catalogs.
  - Type: list | Default: None | Env Var: ``UNITXT_CATALOGS``
- ``unitxt.settings.use_only_local_catalogs``: Restricts the system to using only local catalogs.
  - Type: bool | Default: False | Env Var: ``UNITXT_USE_ONLY_LOCAL_CATALOGS``

Evaluation
----------
- ``unitxt.settings.num_resamples_for_instance_metrics``: Number of bootstrap confidence interval resamples for instance-level metrics.
  - Type: int | Default: 1000 | Env Var: ``UNITXT_NUM_RESAMPLES_FOR_INSTANCE_METRICS``
- ``unitxt.settings.num_resamples_for_global_metrics``: Number of bootstrap confidence interval resamples for global metrics.
  - Type: int | Default: 100 | Env Var: ``UNITXT_NUM_RESAMPLES_FOR_GLOBAL_METRICS``

Debugging
---------
- ``unitxt.settings.use_eager_execution``: Enables eager execution mode for debugging. This mode disables streaming and ensures that each operation processes the entire stream before proceeding to the next step. This can be helpful for debugging by making the execution order more predictable.
  - Type: bool | Default: False | Env Var: ``UNITXT_USE_EAGER_EXECUTION``

Privacy
-------
- ``unitxt.settings.data_classification_policy``: Defines the data classification policy.
  - Type: str | Default: None | Env Var: ``UNITXT_DATA_CLASSIFICATION_POLICY``

