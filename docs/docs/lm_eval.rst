.. _helm:

========================
Running Unitxt with LM-Eval
========================

Unitxt can be integrated with :ref:`LM-Evaluation-Harness <https://github.com/EleutherAI/lm-evaluation-harness>`, enabling you to select and evaluate models from the extensive lm-evaluation-harness models catalog with data recipes created by Unitxt.

First, install lm-evaluation-harness from source (later will be available through a set version)

.. code-block:: bash

    pip install git+git:https://github.com/EleutherAI/lm-evaluation-harness

Next, define your preferred Unitxt recipe:

.. code-block:: bash

    recipe="card=cards.wnli,template=templates.classification.multi_class.relation.default"

If you're unsure about your choice, you can use the :ref:`Explore Unitxt <demo>` tool for an interactive recipe exploration UI. After making your selection, click on Generate Prompts, and then click on the Code tab. You will see a code snippet such as the following:

.. code-block:: python

    dataset = load_dataset('unitxt/data', 'card=cards.wnli,template=templates.classification.multi_class.relation.default,max_train_instances=5', split='train')

The second string parameter to `load_dataset()` is the recipe. Note that you will might want to remove `max_train_instances=5` from the recipe before using it, as the `max_train_instances`. If you wish to use few-shot in-context learning, you should configure this using the `num_demos` and `demos_pool_size` parameters instead e.g. `num_demos=5,demos_pool_size=10`.

Once you have your Unitxt recipe set go and save it in the lm-evaluation-harness `tasks/unitxt` directory.

You can call your task `wnli` and save it under `tasks/unitxt/wnli.yaml` in a YAML file:


.. code-block:: yaml
    task: wnli
    include: unitxt
    recipe: ard=cards.wnli,template=templates.classification.multi_class.relation.default


Select the model you wish to evaluate from the many types of models supported by lm-evaluation-harness platform (for a comprehensive list, refer to: https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#model-apis-and-inference-servers):

Run your newly constructed task with:

.. code-block:: bash

    lm_eval --model hf \
        --model_args pretrained=google/flan-t5-base \
        --tasks wnli



