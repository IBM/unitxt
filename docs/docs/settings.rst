.. _settings:

=====================================
Library Settings and Constants
=====================================

This guide explains the rationale behind the :class:`Settings <settings_utils.Settings>` and :class:`Constants <settings_utils.Constants>` system, how to extend and configure them, and how to use them effectively in your application.

All the settings can be easily accessed with:

.. code-block:: python

    import unitxt

    print(unitxt.settings.default_verbosity)  # Output: "info"

All the settings can be easily modified with:

.. code-block:: python

    unitxt.settings.default_verbosity = "debug"

Or through environment variables:

.. code-block::

    export UNITXT_DEFAULT_VERBOSITY = "debug"

Rationale
=========

Managing application-wide configuration and constants can be challenging, especially in larger systems. The :class:`Settings <settings_utils.Settings>` and :class:`Constants <settings_utils.Constants>` classes provide a centralized, thread-safe, and type-safe way to manage these configurations.

- **Settings**: Designed for mutable configurations that can be customized dynamically, with optional type enforcement and environment variable overrides.
- **Constants**: Designed for immutable values that remain consistent throughout the application lifecycle.

By centralizing these configurations, you can:
- Ensure consistency across your application.
- Simplify debugging and testing.
- Enable dynamic configuration using environment variables or runtime contexts.

Adding New Settings
===================

To add a new setting, follow these steps:

1. Open the :class:`Settings <settings_utils.Settings>` initialization block in the :class:`settings_utils <settings_utils>` module.
2. Add a new setting key with a tuple of `(type, default_value)` to enforce its type and provide a default value.

.. code-block:: python

    settings.new_feature_enabled = (bool, False)  # Adding a new boolean setting.

Guidelines:
- Use a clear and descriptive name for the setting.
- Always specify the type as one of `int`, `float`, or `bool`.

Adding New Constants
====================

To add a new constant:

1. Open the :class:`Constants <settings_utils.Constants>` initialization block in the :class:`settings_utils <settings_utils>` module.
2. Assign a new constant key with its value.

.. code-block:: python

    constants.new_constant = "new_value"  # Adding a new constant.

Guidelines:
- Constants should represent fixed, immutable values.
- Use clear and descriptive names that indicate their purpose.

Using Settings Context
======================

The :class:`Settings <settings_utils.Settings>` class provides a `context` manager to temporarily override settings within a specific block of code. After exiting the block, the settings revert to their original values.

Example:

.. code-block:: python

    from unitxt import settings

    print(settings.default_verbosity)  # Output: "info"

    with settings.context(default_verbosity="debug"):
        print(settings.default_verbosity)  # Output: "debug"

    print(settings.default_verbosity)  # Output: "info"

This feature is useful for scenarios like testing or running specific tasks with modified configurations.

List of Settings
================

Below is the list of available settings, their types, default values, corresponding environment variable names, and descriptions:

.. list-table::
   :header-rows: 1

   * - Setting
     - Type
     - Default Value
     - Environment Variable
     - Description
   * - allow_unverified_code
     - bool
     - False
     - UNITXT_ALLOW_UNVERIFIED_CODE
     - Enables or disables execution of unverified code.
   * - use_only_local_catalogs
     - bool
     - False
     - UNITXT_USE_ONLY_LOCAL_CATALOGS
     - Restricts operations to use only local catalogs.
   * - global_loader_limit
     - int
     - None
     - UNITXT_GLOBAL_LOADER_LIMIT
     - Sets a limit on the number of global data loaders.
   * - num_resamples_for_instance_metrics
     - int
     - 1000
     - UNITXT_NUM_RESAMPLES_FOR_INSTANCE_METRICS
     - Number of resamples used for calculating instance-level metrics.
   * - num_resamples_for_global_metrics
     - int
     - 100
     - UNITXT_NUM_RESAMPLES_FOR_GLOBAL_METRICS
     - Number of resamples used for calculating global metrics.
   * - max_log_message_size
     - int
     - 100000
     - UNITXT_MAX_LOG_MESSAGE_SIZE
     - Maximum size allowed for log messages.
   * - catalogs
     - None
     - None
     - UNITXT_CATALOGS
     - Specifies the catalogs configuration.
   * - artifactories
     - None
     - None
     - UNITXT_ARTIFACTORIES
     - Defines the artifact storage configuration.
   * - default_recipe
     - str
     - "dataset_recipe"
     - UNITXT_DEFAULT_RECIPE
     - Specifies the default recipe for datasets.
   * - default_verbosity
     - str
     - "info"
     - UNITXT_DEFAULT_VERBOSITY
     - Sets the default verbosity level for logging.
   * - use_eager_execution
     - bool
     - False
     - UNITXT_USE_EAGER_EXECUTION
     - Enables eager execution for tasks.
   * - remote_metrics
     - list
     - []
     - UNITXT_REMOTE_METRICS
     - Defines a list of configurations for remote metrics.
   * - test_card_disable
     - bool
     - False
     - UNITXT_TEST_CARD_DISABLE
     - Disables the use of test cards when enabled.
   * - test_metric_disable
     - bool
     - False
     - UNITXT_TEST_METRIC_DISABLE
     - Disables the use of test metrics when enabled.
   * - metrics_master_key_token
     - None
     - None
     - UNITXT_METRICS_MASTER_KEY_TOKEN
     - Specifies the master token for accessing metrics.
   * - seed
     - int
     - 42
     - UNITXT_SEED
     - Default seed value for random operations.
   * - skip_artifacts_prepare_and_verify
     - bool
     - False
     - UNITXT_SKIP_ARTIFACTS_PREPARE_AND_VERIFY
     - Skips preparation and verification of artifacts.
   * - data_classification_policy
     - None
     - None
     - UNITXT_DATA_CLASSIFICATION_POLICY
     - Specifies the policy for data classification.
   * - mock_inference_mode
     - bool
     - False
     - UNITXT_MOCK_INFERENCE_MODE
     - Enables mock inference mode for testing.
   * - disable_hf_datasets_cache
     - bool
     - True
     - UNITXT_DISABLE_HF_DATASETS_CACHE
     - Disables caching for Hugging Face datasets.
   * - loader_cache_size
     - int
     - 1
     - UNITXT_LOADER_CACHE_SIZE
     - Sets the cache size for data loaders.
   * - task_data_as_text
     - bool
     - True
     - UNITXT_TASK_DATA_AS_TEXT
     - Enables representation of task data as plain text.
   * - default_provider
     - str
     - "watsonx"
     - UNITXT_DEFAULT_PROVIDER
     - Specifies the default provider for tasks.
   * - default_format
     - None
     - None
     - UNITXT_DEFAULT_FORMAT
     - Defines the default format for data processing.

List of Constants
=================

Below is the list of available constants and their values:

.. list-table::
   :header-rows: 1

   * - Constant
     - Value
   * - dataset_file
     - Path to `dataset.py`.
   * - metric_file
     - Path to `metric.py`.
   * - local_catalog_path
     - Path to the local catalog directory.
   * - package_dir
     - Directory of the installed package.
   * - default_catalog_path
     - Default catalog directory path.
   * - dataset_url
     - URL for dataset resources.
   * - metric_url
     - URL for metric resources.
   * - version
     - Current version of the application.
   * - catalog_hierarchy_sep
     - Separator for catalog hierarchy levels.
   * - env_local_catalogs_paths_sep
     - Separator for local catalog paths in environment variables.
   * - non_registered_files
     - List of files excluded from registration.
   * - codebase_url
     - URL of the codebase repository.
   * - website_url
     - Official website URL.
   * - inference_stream
     - Name of the inference stream constant.
   * - instance_stream
     - Name of the instance stream constant.
   * - image_tag
     - Default image tag for operations.
   * - demos_pool_field
     - Field name for demos pool.

Conclusion
==========

The `Settings` and `Constants` system provides a robust and flexible way to manage your application's configuration and constants. By following the guidelines above, you can extend and use these classes effectively in your application.