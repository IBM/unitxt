.. _adding_template:

.. note::

   To use this tutorial, you need to :ref:`install unitxt <install_unitxt>`.


=====================================
Adding Templates ✨
=====================================

In this section you learn how to add a Template. Templates are the way for unitxt to take your task data and verbalize the task instructions to the model.
The templates made by the community can be found in the catalog :ref:`templates section <catalog.templates>`.
And the documentation for the base classes used for templates can be found here: :ref:`Templates Documentation<templates>`

.. image:: ../../assets/unitxt_flow.png
   :alt: The unitxt flow
   :width: 100%
   :align: center

Adding a new Template
----------------------------


.. code-block:: python

    ..
    templates=TemplatesList([
        InputOutputTemplate(
            instruction="In the following task you translate a {text_type}."
            input_format="Translate this {text_type} from {source_language} to {target_language}: {text}.",
            target_prefix="Translation: ",
            output_format='{translation}',
        ),
    ])