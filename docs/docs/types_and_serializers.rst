.. _types_and_serializers:

=====================================
Types and Serializers
=====================================

Unitxt employs various tools for serializing data into textual format. One of these mechanisms is the Type-Serializers mechanism, which assigns serialization to specific types. For example, consider the following ``typing`` types:

.. code-block:: python

    from typing import NewType, TypedDict, Union, Literal, List, Any

    Text = NewType("Text", str)
    Number = NewType("Number", Union[float, int])

    class Turn(TypedDict):
        role: Literal["system", "user", "agent"]
        content: Text

    Dialog = NewType("Dialog", List[Turn])

    class Table(TypedDict):
        header: List[str]
        rows: List[List[Any]]

For each type, we can assign different serialization methods specific to that type. This enables us to "plug and play" different serialization methods and modify our data’s textual representation accordingly.

Registering the Types
-----------------------

First, we need to register the different types we want to support from the Python ``typing`` types defined above.

.. code-block:: python

    from unitxt.type_utils import register_type

    register_type(Text)
    register_type(Number)
    register_type(Turn)
    register_type(Dialog)
    register_type(Table)

Defining a Serializer for a Type
---------------------------------

Once the types are registered, we can define serializers for those types. For example, consider creating a serializer for the ``Dialog`` type:

.. code-block:: python

    from unitxt.serializers import SingleTypeSerializer

    class DialogSerializer(SingleTypeSerializer):
        serialized_type = Dialog

        def serialize(self, value: Dialog, instance: Dict[str, Any]) -> str:
            # Convert the Dialog into a string representation, typically combining roles and content
            return "\n".join(f"{turn['role']}: {turn['content']}" for turn in value)

Using the New Serializer
--------------------------

To use the new serializer, we need to do two things:
1. Ensure our task supports this type.
2. Add the serializer to the data loading recipe.

Using the New Type in a Task
----------------------------

Once the new type is registered, we can create a task that requires this type:

.. code-block:: python

    from unitxt.task import Task

    dialog_summarization_task = Task(
        input_fields={"dialog": Dialog},
        reference_fields={"summary": str},
        prediction_type=str,
        metrics=["metrics.rouge"],
    )

Loading Data with the Serializer
---------------------------------

Once the task is defined with the type, we can use the serializer when loading the data.

Given this standalone card:

.. code-block:: python

    data = {
        "test": [
            {
                "dialog": [
                    {"role": "user", "content": "What is the time?"},
                    {"role": "system", "content": "4:13 PM"},
                ],
                "summary": "User asked for the time and got an answer."
            }
        ]
    }

    card = TaskCard(
        loader=LoadFromDictionary(data=data),
        task=dialog_summarization_task,
    )

We can load the data with the serializer using:

.. code-block:: python

    dataset = load_dataset(
        card=card,
        template=InputOutputTemplate(
            instruction="Summarize the following dialog.",
            input_format="{dialog}",
            output_format="{summary}",
        ),
        serializer=DialogSerializer(),
    )

Now if you print the input of the first instance of the dataset by ``print(dataset["test"][0]["source"])`` you will get:

.. code-block::

    Summarize the following dialog.
    user: What is the time?
    system: 4:13 PM



Adding a Serializer to a Template
------------------------------------

Another option is to set a default serializer for a given template per field. 
.. code-block:: python

    from unitxt.serializers import (
        DialogSerializer
    )

Now, we can add them to the template:

.. code-block:: python

    InputOutputTemplate(
        instruction="Summarize the following dialog.",
        input_format="{dialog}",
        output_format="{summary}",
        serializer={"dialog": DialogSerializer}
    )

As another example, we can use a customized ListSerializer, to format list of contexts with 
two newlines separators, instead of the standard comma list separator. 

.. code-block:: python

    from unitxt.serializers import (
        ListSerializer
    )
    template = MultiReferenceTemplate(
        instruction="Answer the question based on the information provided in the document given below.\n\n",
        input_format="Contexts:\n\n{contexts}\n\nQuestion: {question}",
        references_field="reference_answers",
        serializer={"contexts": ListSerializer(separator="\n\n")},
    )

