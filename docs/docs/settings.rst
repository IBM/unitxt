.. _settings:

=====================================
Library Settings and Constants
=====================================

This guide explains the rationale behind the :class:`Settings <settings_utils.Settings>` and :class:`Constants <settings_utils.Constants>` system, how to extend and configure them, and how to use them effectively in your application.

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

Below is the list of available settings, their types, and default values:

.. list-table::
   :header-rows: 1

   * - Setting
     - Type
     - Default Value
   * - allow_unverified_code
     - bool
     - False
   * - use_only_local_catalogs
     - bool
     - False
   * - global_loader_limit
     - int
     - None
   * - num_resamples_for_instance_metrics
     - int
     - 1000
   * - num_resamples_for_global_metrics
     - int
     - 100
   * - max_log_message_size
     - int
     - 100000
   * - catalogs
     - None
     - None
   * - artifactories
     - None
     - None
   * - default_recipe
     - str
     - "dataset_recipe"
   * - default_verbosity
     - str
     - "info"
   * - use_eager_execution
     - bool
     - False
   * - remote_metrics
     - list
     - []
   * - test_card_disable
     - bool
     - False
   * - test_metric_disable
     - bool
     - False
   * - metrics_master_key_token
     - None
     - None
   * - seed
     - int
     - 42
   * - skip_artifacts_prepare_and_verify
     - bool
     - False
   * - data_classification_policy
     - None
     - None
   * - mock_inference_mode
     - bool
     - False
   * - disable_hf_datasets_cache
     - bool
     - True
   * - loader_cache_size
     - int
     - 1
   * - task_data_as_text
     - bool
     - True
   * - default_provider
     - str
     - "watsonx"
   * - default_format
     - None
     - None

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