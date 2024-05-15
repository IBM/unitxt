.. _adding_operator:

.. note::

   To use this tutorial, you need to :ref:`install unitxt <install_unitxt>`.


=====================================
Operators ✨
=====================================

In this section we will add brand new stream operators to review the existing operators  :ref:`press here <operators>`.

Adding a new stream operator
----------------------------

Create a new class that extends the `StreamInstanceOperator` or any other :ref:`Stream Operator <operators>` class.

    .. code-block:: python

        from unitxt.operator import StreamInstanceOperator

        class AddFields(StreamInstanceOperator):
            fields: Dict[str, object]

            def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
                return {**instance, **self.fields}

To test that our operator works as expected, we can use the unitxt built-in
testing suit:

    .. code-block:: python

        from unitxt.test_utils.operators import check_operator

        operator = AddFields(fields={"b": 2})

        inputs = [{'a': 1}, {'a': 2}]
        targets = [{'a': 1, 'b': 2}, {'a': 2, 'b': 2}]

        print(check_operator(operator, inputs, targets)) # True

