========================
Running Unitxt with HELM
========================

.. _prompt_format_layout:
.. image:: ../../assets/helm+unitxt.png
   :alt: HELM + Unitxt
   :width: 75%
   :align: center

Unitxt can be integrated with HELM, enabling you to select and evaluate models from the extensive HELM models catalog with data recipes created by Unitxt.

First, install HELM at version v0.5.0 or later:

.. code-block:: bash

    pip install --upgrade 'crfm-helm[unitxt]>=0.5.0'

Next, define your preferred Unitxt recipe:

.. code-block:: bash

    recipe="card=cards.wnli,template=templates.classification.multi_class.relation.default"

If you're unsure about your choice, consider using the `unitxt-explore` tool for an interactive recipe exploration UI.

Select the model you wish to evaluate from the HELM catalog (for a comprehensive list, refer to: https://crfm-helm.readthedocs.io/en/latest/models/):

.. code-block:: bash

    model="openai/gpt2"

To execute the evaluation, combine the components with the following command:

.. code-block:: bash

    helm-run \
        --run-entries "unitxt:$recipe,model=$model" \
        --max-eval-instances 10 --suite v1

Unitxt also supports evaluating models available on the Hugging Face Hub:

.. code-block:: bash

    model="stanford-crfm/alias-gpt2-small-x21"
    helm-run \
        --run-entries "unitxt:$recipe,model=$model" \
        --enable-huggingface-models $model \
        --max-eval-instances 10 --suite v1

To summarize the results of all runs within the created suite, use:

.. code-block:: bash

    helm-summarize --suite v1

To view the aggregated results look at `benchmark_output/runs/v1/unitxt:$recipe,model=${model/\//_}/stats.json`

Finally, to review the predictions in your web browser, execute:

.. code-block:: bash

    helm-server


