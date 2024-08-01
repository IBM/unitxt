.. _adding_format:

.. note::

   To use this tutorial, you need to :ref:`install unitxt <install_unitxt>`.

=====================================
Formats ✨
=====================================

Formats define the overall textual layout of the example, including system prompt,
in-context learning demonstrations, and other special tokens.
The format and template work together to verbalize the model input -
the template verbalizes the task specific parts of the input prompt
while the format verbalizes the model specific aspects of the input prompt.

In-context learning is activated when the  ``num_demos`` parameter of
the :ref:`recipe <recipe>` is set to a non zero value.
Different demo examples are chosen per instance from a fixed set of examples called a ``demo_pool``.
Usually, the examples in the demo pool are taken from the train split, but this can be overridden by the ``demos_taken_from`` parameter.
The size of the demo pool is determined by a mandatory parameter called ``demos_pool_size`` parameter.


.. _prompt_format_layout:
.. image:: ../../assets/prompt_layout.png
   :alt: The unitxt prompt layout
   :width: 75%
   :align: center

It determines the positioning of the task `instruction`, `system_prompt` and `demos` the `source` query and required output from the model, the `target`.

Below is in example of how to define the layout of the different parts.
This example is based on this blog post explaining the prompt structure of the llama2 model: `Blog Post<https://huggingface.co/blog/llama2#how-to-prompt-llama-2>`_

So the actual template looks like this:

.. Example
::

   <s>[INST] <<SYS>>
   {{ system_prompt }}
   <</SYS>>

   {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]

An example for data point with the llama2 format and system prompt.

.. Example
::

   [INST] <<SYS>>
   You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.


   If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.


   Classify the grammatical acceptability of the following text to one of these options: unacceptable, acceptable.
   <<SYS>>


   text: The more we study verbs, the crazier they get. [/INST] The grammatical acceptability is acceptable </s><s>[INST] text: They drank the pub. [/INST]The grammatical acceptability is

To define this exact format you can use this code:

.. code-block:: python

   from unitxt.catalog import add_to_catalog
   from unitxt.formats import SystemFormat

   format = SystemFormat(
      demo_format="{source} [/INST] {target_prefix}{target} </s><s>[INST] ",
      model_input_format="[INST] <<SYS>>\n{system_prompt}\n\n{instruction}<<SYS>>\n\n\n{demos}{source} [/INST]{target_prefix}",
   )

   add_to_catalog(format, "formats.llama2", overwrite=True)
