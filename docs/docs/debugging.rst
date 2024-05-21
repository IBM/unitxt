.. _debugging:

===================================
Debugging Unitxt
===================================

Increase log verbosity
-----------------

If you want to get more information during the run (for example, which artifict are loaded from which catalog),
you can set the UNITXT_DEFAULT_VERBOSITY environment variable or modify the global setting in the code.

.. code-block:: bash

  env UNITXT_DEFAULT_VERBOSITY=debug python prepare/cards/wnli.py

.. code-block:: python

  from .settings_utils import get_settings
  settings = get_settings()
  settings.default_verbosity = "debug"