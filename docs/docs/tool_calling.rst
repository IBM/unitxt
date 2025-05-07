.. _tool_calling:

===========
Tool Calling
===========

.. note::

   This tutorial requires a :ref:`Unitxt installation <install_unitxt>`.

Introduction
------------

This tutorial explores tool calling with Unitxt, focusing on handling tool-based datasets and creating an evaluation and inference pipeline. By the end, you'll be equipped to process complex tool calling tasks efficiently.

Part 1: Understanding Tool Calling Tasks
---------------------------------------

Tool calling tasks involve providing a model with instructions or prompts that require the use of specific tools to generate the correct response. These tasks are increasingly important in modern AI applications that need to interact with external systems.

Tool Calling Schema
^^^^^^^^^^^^^^^^^

Unitxt uses specific typed structures for tool calling:

.. code-block:: python

    class Tool(TypedDict):
        name: str
        description: str
        parameters: JsonSchema # a well defined json schema

    class ToolCall(TypedDict):
        name: str
        arguments: Dict[str, Any]

The task schema for supervised tool calling is defined as:

.. code-block:: python

    Task(
        __description__="""Task to test tool calling capabilities.""",
        input_fields={"query": str, "tools": List[Tool]},
        reference_fields={"call": ToolCall},
        prediction_type=ToolCall,
        metrics=["metrics.tool_calling"],
        default_template="templates.tool_calling.base",
    )

This schema appears in the catalog as ``tasks.tool_calling.supervised`` and is the foundation for our tool calling evaluation pipeline.

Tutorial Overview
^^^^^^^^^^^^^^^^^

We'll create a tool calling evaluation pipeline using Unitxt, concentrating on tasks where models need to select the right tool and provide appropriate arguments for the tool's parameters. We'll use the Berkeley Function Calling Leaderboard as our example dataset.

Part 2: Data Preparation
-----------------------

Creating a Unitxt DataCard
^^^^^^^^^^^^^^^^^^^^^^^^^

Our first step is to prepare the data using a Unitxt DataCard. If it's your first time adding a DataCard, we recommend reading the :ref:`Adding Datasets Tutorial <adding_dataset>`.

Dataset Selection
^^^^^^^^^^^^^^^

We'll use the Berkeley Function Calling Leaderboard dataset, which is designed to evaluate LLMs' ability to call functions correctly across diverse categories and use cases.

DataCard Implementation
^^^^^^^^^^^^^^^^^^^^^

Create a Python file named ``bfcl.py`` and implement the DataCard as follows:

.. code-block:: python

    import unitxt
    from unitxt.card import TaskCard
    from unitxt.catalog import add_to_catalog
    from unitxt.collections_operators import DictToTuplesList, Wrap
    from unitxt.loaders import LoadCSV
    from unitxt.operators import Copy
    from unitxt.stream_operators import JoinStreams
    from unitxt.test_utils.card import test_card
    from unitxt.tool_calling import ToTool

    # Base path to the Berkeley Function Calling Leaderboard data
    base_path = "https://raw.githubusercontent.com/ShishirPatil/gorilla/70b6a4a2144597b1f99d1f4d3185d35d7ee532a4/berkeley-function-call-leaderboard/data/"

    with unitxt.settings.context(allow_unverified_code=True):
        card = TaskCard(
            loader=LoadCSV(
                files={"questions": base_path + "BFCL_v3_simple.json", "answers": base_path + "possible_answer/BFCL_v3_simple.json"},
                file_type="json",
                lines=True,
                data_classification_policy=["public"],
            ),
            preprocess_steps=[
                # Join the questions and answers streams
                JoinStreams(left_stream="questions", right_stream="answers", how="inner", on="id", new_stream_name="test"),
                # Extract the query from the question content
                Copy(field="question/0/0/content", to_field="query"),
                # Starting to build the tools field as List[Tool]
                Copy(field="function", to_field="tools"),
                # Make Sure the json schema of the parameters is well defined
                RecursiveReplace(key="type", map_values={"dict": "object", "float": "number", "tuple": "array"}, remove_values=["any"]),
                # Process ground truth data
                DictToTuplesList(field="ground_truth/0", to_field="call_tuples"),
                # Extract tool name and arguments from ground truth
                Copy(field="call_tuples/0/0", to_field="call/name"),
                Copy(field="call_tuples/0/1", to_field="call/arguments"),
            ],
            task="tasks.tool_calling.supervised",
            templates=["templates.tool_calling.base"],
            __description__=(
                """The Berkeley function calling leaderboard is a live leaderboard to evaluate the ability of different LLMs to call functions (also referred to as tools). We built this dataset from our learnings to be representative of most users' function calling use-cases, for example, in agents, as a part of enterprise workflows, etc. To this end, our evaluation dataset spans diverse categories, and across multiple languages."""
            ),
        )

        # Test and add the card to the catalog
        test_card(card, strict=False)
        add_to_catalog(card, "cards.bfcl.simple_v3", overwrite=True)

Preprocessing for Task Schema
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each preprocessing step serves a specific purpose in transforming the raw data into the required task schema:

1. ``JoinStreams``: Combines question and answer data based on ID
2. ``Copy(field="question/0/0/content", to_field="query")``: Creates the ``query`` input field
3. ``Copy(field="function", to_field="tools")``: Creates the ``tools`` list input field
4. ``RecursiveReplace(key="type", map_values={"dict": "object", "float": "number", "tuple": "array"}, remove_values=["any"])``: Converts parameters definitions to the ``JsonSchema`` structure
5. ``DictToTuplesList`` and subsequent ``Copy`` operations: Create the reference ``call`` field with the proper ``ToolCall`` structure

After preprocessing, each example will have:
- A ``query`` that the model should respond to
- Available ``tools`` that the model can choose from
- A reference ``call`` showing which tool should be called with what arguments

Part 3: Inference and Evaluation
-------------------------------

With our data prepared, we can now test model performance on tool calling tasks.

Pipeline Setup
^^^^^^^^^^^^^

Set up the inference and evaluation pipeline:

.. code-block:: python

    from unitxt import get_logger
    from unitxt.api import evaluate, load_dataset
    from unitxt.inference import CrossProviderInferenceEngine

    logger = get_logger()

    # Load and prepare the dataset
    dataset = load_dataset(
        card="cards.bfcl.simple_v3",
        split="test",
        format="formats.chat_api",  # Format suitable for tool calling
    )

    # Initialize the inference model with a compatible provider
    model = CrossProviderInferenceEngine(
        model="granite-3-3-8b-instruct",  # Or other models supporting tool calling
        provider="watsonx"
    )

Executing Inference and Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the model and evaluate the results:

.. code-block:: python

    # Perform inference
    predictions = model(dataset)

    # Evaluate the predictions
    results = evaluate(predictions=predictions, data=dataset)

    # Print the results
    print("Global Results:")
    print(results.global_scores.summary)

    print("Instance Results:")
    print(results.instance_scores.summary)

Part 4: Understanding the Tool Calling Metrics
--------------------------------------------

The ToolCallingMetric in Unitxt provides several useful scores:

.. code-block:: python

    class ToolCallingMetric(ReductionInstanceMetric[str, Dict[str, float]]):
        main_score = "exact_match"
        reduction = MeanReduction()
        prediction_type = ToolCall

        def map(
            self, prediction: ToolCall, references: List[ToolCall], task_data: Dict[str, Any]
        ) -> Dict[str, float]:
            # Implementation details...
            return {
                self.main_score: exact_match,
                "tool_choice": tool_choice,
                "parameter_choice": parameter_choice,
                "parameters_types": parameters_types,
                "parameter_values": parameter_values
            }

The metrics evaluate different aspects of tool calling accuracy:

1. **exact_match**: Measures if the tool call exactly matches a reference
2. **tool_choice**: Evaluates if the correct tool was selected
3. **parameter_choice**: Checks if the correct parameters were identified
4. **parameter_values**: Assesses if the parameter values are correct
5. **parameters_types**: Verifies if parameter types match the tool definition

Custom Evaluation
^^^^^^^^^^^^^^^

For more specialized evaluation, you can define custom metrics:

.. code-block:: python

    from unitxt.metrics import ToolCallingMetric

    # Evaluate with a specialized tool calling metric
    custom_results = evaluate(
        predictions=predictions,
        data=dataset,
        metrics=[ToolCallingMetric()]
    )

    print("Custom Metric Results:")
    print(custom_results.global_scores.summary)

Example Analysis
^^^^^^^^^^^^^^^

To better understand your model's performance, analyze individual instances:

.. code-block:: python

    # Display detailed results for the first few instances
    for i, instance in enumerate(results.instance_scores.data[:3]):
        print(f"\nInstance {i+1}:")
        print(f"Query: {dataset[i]['query']}")
        print(f"Available tools: {dataset[i]['tools']}")
        print(f"Expected tool call: {dataset[i]['call']}")
        print(f"Model prediction: {predictions[i]}")
        print(f"Scores: {instance}")

Testing with Different Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can easily compare different models' performance:

.. code-block:: python

    # Test with a different model
    alternative_model = CrossProviderInferenceEngine(
        model="gpt-3.5-turbo",
        provider="openai"
    )

    alt_predictions = alternative_model(dataset)
    alt_results = evaluate(predictions=alt_predictions, data=dataset)

    print("Alternative Model Results:")
    print(alt_results.global_scores.summary)

Conclusion
---------

You have now successfully implemented a tool calling evaluation pipeline with Unitxt using the Berkeley Function Calling Leaderboard dataset. This capability enables the assessment of models' ability to use tools correctly, opening up new possibilities for AI applications that interact with external systems.

The structured approach using typed definitions (``Parameter``, ``Tool``, and ``ToolCall``) provides a standardized way to evaluate tool calling capabilities across different models and providers.

We encourage you to explore further by experimenting with different datasets, models, and evaluation metrics to fully leverage Unitxt's capabilities in tool calling assessment.