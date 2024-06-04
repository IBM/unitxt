.. _adding_operator:

.. note::

   To use this tutorial, you need to :ref:`install unitxt <install_unitxt>`.

=====================================
Operators ✨
=====================================

Operators are specialized functions designed to process data.

They are used in the TaskCard for preparing data for specific tasks and by Post Processors
to process the textual output of the model to the expect input of the metrics. 

There are several types of operators. 

1. Field Operators - Operators that modify individual fields of the instances in the input streams.  Example of such operators are operators that
cast field values, uppercase string fields, or translate text between languages.

2. Instance Operators - Operators that modify individual instances in the input streams. For example, operators that add or remove fields.

3. Stream Operators - Operators that perform operations on full streams. For example, operators that remove instances based on some condition.

4. MultiStream Operators - Operator that perform operations on multiple streams.  For example, operators that repartition the instances between train and test splits.

Unitxt comes with a large collection of built in operators - that were design to cover most common requirements of dataset processing.

The list of available operators can be found in the :ref:`operators <operators>` section.

Built in operators have some benefits:

1. **Testability**: Built-in Operators are unit-tested, ensuring reliability and functionality.
2. **Code Reusability**: Shared, well-maintained operator code can be reused across projects.
3. **Performance**: Built-in Operators are designed to maintain high performance standards, particularly suitable for stream processing.
4. **Security** : Built-in Operators do not require running of arbitrary user code, and hence can be run in secured environments that prohibit running user code.

It is recommended to use existing operators when possible. 

However, if a highly specific or uncommon operation is needed that existing operators do not cover, and it is unlikely to be reused, you can use :ref:`ExecuteExpression <operators.ExecuteExpression>`  or :ref:`FilterByExpression <operators.FilterByExpression>`operators:

.. code-block:: python

    ExecuteExpression('question + "?"', to_field="question")
    FilterByExpression('len(question) == 0')

**Explanation**: These lines demonstrate how to use two specific operators for string manipulation and conditional filtering.

In addition, this tutorial will now guide you on creating new operators in Python for personal use and community contribution.

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

Stream Operators
----------------

Stream operators are designed to manage and manipulate entire data streams. These operators process instances sequentially, allowing for operations that affect the entire stream, such as limiting the number of instances processed.

.. code-block:: python

    from unitxt.stream import Stream
    from unitxt.operator import StreamOperator

    class LimitSize(StreamOperator):
        size: int
        def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
            for i, instance in enumerate(stream):
                if i > self.size:
                    break
                yield instance

**Explanation**: The `LimitSize` class inherits from `StreamOperator` and is used to limit the number of instances processed in a stream. It iterates over each instance in the stream and stops yielding new instances once the specified size limit is exceeded. This operator is useful for scenarios such as data sampling or when resource constraints limit the number of instances that can be processed.

MultiStream Operators
---------------------

MultiStream operators handle operations across multiple data streams concurrently. These operators are capable of merging, filtering, or redistributing data from multiple streams into a new stream configuration.

.. code-block:: python

    from unitxt.stream import MultiStream, GeneratorStream
    from unitxt.operator import MultiStreamOperator

    class MergeAllStreams(MultiStreamOperator):

        def merge(self, streams) -> Generator:
            for stream in streams:
                for instance in stream:
                    yield instance

        def process(self, multi_stream: MultiStream) -> MultiStream:
            return MultiStream(
                {
                    "merged": GeneratorStream(
                        self.merge, gen_kwargs={"streams": multi_stream.values()}
                    )
                }
            )

**Explanation**: The `MergeAllStreams` class extends `MultiStreamOperator` and provides functionality to merge several streams into a single stream. 
The `merge` method iterates over each provided stream, yielding instances from each one consecutively. The `process` method then utilizes this merging logic to create a new `MultiStream` that consolidates all input streams into a single output stream named "merged". 
This operator is particularly useful in scenarios where data from different sources needs to be combined into a single dataset for analysis or further processing.

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
