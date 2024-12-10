.. _lm-eval:

===========================
Running Unitxt with LM-Eval
===========================

Unitxt can be seamlessly integrated with the `LM-Evaluation-Harness <https://github.com/EleutherAI/lm-evaluation-harness>`_, enabling the selection and evaluation of models from the extensive lm-evaluation-harness models catalog using data recipes created by Unitxt.

Installation
------------

To begin, install lm-evaluation-harness from the source (a set version will be available in the future):

.. code-block:: bash

    pip install git+https://github.com/EleutherAI/lm-evaluation-harness

Define Your Unitxt Recipe
-------------------------

Next, choose your preferred Unitxt recipe:

.. code-block:: python

    card=cards.wnli,template=templates.classification.multi_class.relation.default

If you are uncertain about your choice, you can utilize the :ref:`Explore Unitxt <demo>` tool for an interactive recipe exploration UI. After making your selection, click on "Generate Prompts," and then navigate to the "Code" tab. You will see a code snippet similar to the following:

.. code-block:: python

    dataset = load_dataset('unitxt/data', 'card=cards.wnli,template=templates.classification.multi_class.relation.default,max_train_instances=5', split='train')

The second string parameter to `load_dataset()` is the recipe. Note that you may want to remove `max_train_instances=5` from the recipe before using it. If you wish to employ few-shot in-context learning, configure this using the `num_demos` and `demos_pool_size` parameters instead, e.g., `num_demos=5,demos_pool_size=10`.

Set Up Your Custom LM-Eval Unitxt Tasks Directory
-------------------------------------------------

First, create a directory:

.. code-block:: bash

    mkdir ./my_tasks

Next, run the following code to save the Unitxt configuration file in your tasks directory:

.. code-block:: bash

    python -c 'from lm_eval.tasks.unitxt import task; import os.path; print("class: !function " + task.__file__.replace("task.py", "task.Unitxt"))' > ./my_tasks/unitxt

You will now have a `unitxt` file in your `./my_tasks` directory that defines the integration with your local virtual environment. This step should be performed once. Note that when changing virtual environments, you will need to update it using the code above.

You can designate your task as `my_task` and save it in any folder as `./my_tasks/my_task.yaml` in a YAML file:

.. code-block:: yaml

    task: my_task
    include: unitxt
    recipe: card=cards.wnli,template=templates.classification.multi_class.relation.default

Select the model you wish to evaluate from the diverse types of models supported by the lm-evaluation-harness platform (See a comprehensive list `here <https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#model-apis-and-inference-servers>`_).

Execute your newly constructed task with:

.. code-block:: bash

    lm_eval --model hf \
        --model_args pretrained=google/flan-t5-base \
        --device cpu --tasks my_task --include_path ./my_tasks
