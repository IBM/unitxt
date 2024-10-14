=====================================
Tags and Descriptions
=====================================

Unitxt employ different tools for serializing data into textual
format. One of this mechanism is the Type-Serializers mechanism which assign serialization to specific types.
For example consider the following ``typing types``:

.. code-block:: python

    Text = NewType("Text", str)
    Number = NewType("Number", Union[float, int])

    class Turn(TypedDict):
        role: Literal["system", "user", "agent"]
        content: Text

    Dialog = NewType("Dialog", List[Turn])

    class Table(TypedDict):
        header: List[str]
        rows: List[List[Any]]

For each type we can assign different serialization method specific for that type.
In turn we can plug and play different serialization methods and change our data textual representation accordingly.

Registering the types
---------------------------------------

First we need to register the different types we want to support
from the python ``typing`` types defined above.

.. code-block:: python

    from unitxt.type_utils import register_type

    register_type(Text)
    register_type(Number)
    register_type(Turn)
    register_type(Dialog)
    register_type(Table)

Defining Serializer for a Type
---------------------------------------

Once the types are registered we can define serializers for those types.
For example consider making serializer to the ``Dialog`` type

.. code-block:: python

    from unitxt.serializers import SingleTypeSerializer

    class DialogSerializer(SingleTypeSerializer):
        serialized_type = Dialog

        def serialize(self, value: Dialog, instance: Dict[str, Any]) -> str:
            # Convert the Dialog into a string representation, typically combining roles and content
            return "\n".join(f"{turn['role']}: {turn['content']}" for turn in value)


Using The New Serializer
---------------------------------------

In order to use the new serializer we need to do two things:
(1) make sure our task support this type
(2) add the serializer to the data loading recipe

Using new type in task
-------------------------

Once the new type is regisetered we can create a task that require this type:

.. code-block:: python

    from unitxt.task import Task

    dialog_summarization_task = Task(
        input_fields={"dialog": Dialog},
        reference_fields={"summary": str},
        prediction_type=str,
        metrics=["metrics.rouge"],
    )

Loading the data with the serializer
------------------------------------

Once the task is defined with the type we can use the serializer when loading the data.

Given this stand alone card:

.. code-block:: python

    data = {
        "test": [
            {
                "dialog": [
                    {"role": "user", "content": "what is the time?"},
                    {"role": "system", "content": "4:13pm"},
                ],
                "summary": "User asked for the time and got an answer."
        ]
    }

    card = TaskCard(
        loader=LoadFromDictionary(data=data),
        task=dialog_summarization_task,
    )

We can load the data with the serializer by using:

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
