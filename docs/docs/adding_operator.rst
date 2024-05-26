.. _adding_operator:

.. note::

   To use this tutorial, you need to :ref:`install unitxt <install_unitxt>`.

=====================================
Operators ✨
=====================================

Operators are specialized functions designed to process each instance in your data.
They are particularly useful in preparing data for specific tasks, or for modifying the output of models to fit specific metrics.

Why Use Operators?
------------------

While Python code can directly process data, using operators provides several benefits:

1. **Testability**: Operators are often unit-tested, ensuring reliability and functionality.
2. **Code Reusability**: Shared, well-maintained operator code can be reused across projects.
3. **Performance**: Operators are designed to maintain high performance standards, particularly suitable for stream processing.

However, if a highly specific or uncommon operation is needed that existing operators do not cover, and it is unlikely to be reused, you can use :ref:`ExecuteExpression <operators.ExecuteExpression>`  or :ref:`FilterByExpression <operators.FilterByExpression>`operators:

.. code-block:: python

    ExecuteExpression('question + "?"', to_field="question")
    FilterByExpression('len(question) == 0')

**Explanation**: These lines demonstrate how to use two specific operators for string manipulation and conditional filtering.

It is recommended to use existing operators to encourage code sharing and common code testing. The list of available operators can be found in the :ref:`operators <operators>` section. This tutorial will guide you through creating new operators for personal use and community contribution.

Operators are categorized based on their operation scope, some operate on a specific field of an instance, while others handle the entire instance or multiple instances at once.

Field Operators
---------------

To manipulate a single field, inherit from :class:`FieldOperator <operator.FieldOperator>` and implement your manipulation in the `process` method:

.. code-block:: python

    from unitxt.operator import FieldOperator

    class AddNumber(FieldOperator):
        number: float

        def process(self, value):
            return value + self.number

**Explanation**: This class adds a specified number to the input value. It inherits from `FieldOperator` which is designed to operate on a single field.

Usage example:

.. code-block:: python

    operator = AddNumber(number=5, field="price", to_field="new_price")

**Explanation**: This creates an instance of `AddNumber` to add 5 to the `price` field and store the result in `new_price`.

.. note::

    Every :class:`Operator <operator.Operator>` has a `process_instance` function that can be used for debugging. For example, using `AddNumber` implemented above:

    .. code-block:: python

        operator.process_instance({"price": 0.5})
        # Output: {"price": 0.5, "new_price": 5.5}

**Explanation**: This example demonstrates how to debug the `AddNumber` operator by manually processing a sample instance.

Instance Operators
-------------------

Instance operators process data instance by instance. You can access and manipulate the entire instance directly:

.. code-block:: python

    from unitxt.operator import InstanceOperator

    class Join(InstanceOperator):
        fields: List[str]
        separator: str = ""
        to_field: str

        def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
            instance[self.to_field] = self.separator.join([instance[field] for field in self.fields])
            return instance

**Explanation**: This operator joins multiple fields into a single string, separated by a specified delimiter, and stores the result in another field.

Usage example:

.. code-block:: python

    operator = Join(fields=["title", "text"], separator="\n", to_field="context")

**Explanation**: This operator instance will concatenate the `title` and `text` fields with a newline and store the result in `context`.

Example command output:

.. code-block:: python

    operator.process_instance({"title": "Hello!", "text": "World!"})
    # Output: {"title": "Hello!", "text": "World!", "context": "Hello!\nWorld!"}

**Explanation**: This shows the output of the `Join` operator when processing a sample instance.

Unit Testing Operators
-----------------------

To ensure that an operator functions as expected, it's essential to test it. Here’s how you can use the built-in testing suite in Unitxt:

.. code-block:: python

    from unitxt.test_utils.operators import check_operator

    operator = AddNumber(number=2)  # Assuming AddNumber is already defined
    inputs = [{'price': 100}, {'price': 150}]
    targets = [{'price': 100, 'new_price': 102}, {'price': 150, 'new_price': 152}]

    result = check_operator(operator, inputs, targets)
    print(result)  # Output: True if the operator performs as expected

**Explanation**: This test verifies that the `AddNumber` operator correctly adds 2 to the `price` field and stores the result in `new_price`. The function `check_operator` compares the output against the expected `targets` to confirm correct behavior.
