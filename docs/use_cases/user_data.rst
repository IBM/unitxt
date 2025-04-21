============================================
Evaluate standard tasks with my data
============================================

In Unitxt you can easily evaluate any model on your data:

.. code-block:: python

    # Import required components
    from unitxt import evaluate, create_dataset
    from unitxt.blocks import Task, InputOutputTemplate
    from unitxt.inference import HFAutoModelInferenceEngine

    # Question-answer dataset
    data = [
        {"question": "What is the capital of Texas?", "answer": "Austin"},
        {"question": "What is the color of the sky?", "answer": "Blue"},
    ]

    # Define the task and evaluation metric
    task = Task(
        input_fields={"question": str},
        reference_fields={"answer": str},
        prediction_type=str,
        metrics=["metrics.accuracy"],
    )

    # Create a template to format inputs and outputs
    template = InputOutputTemplate(
        instruction="Answer the following question.",
        input_format="{question}",
        output_format="{answer}",
        postprocessors=["processors.lower_case"],
    )

    # Prepare the dataset
    dataset = create_dataset(
        task=task,
        template=template,
        format="formats.chat_api",
        test_set=data,
        split="test",
    )

    # Set up the model (supports Hugging Face, WatsonX, OpenAI, etc.)
    model = HFAutoModelInferenceEngine(
        model_name="Qwen/Qwen1.5-0.5B-Chat", max_new_tokens=32
    )

    # Generate predictions and evaluate
    predictions = model(dataset)
    results = evaluate(predictions=predictions, data=dataset)

    # Print results
    print("Global Results:\n", results.global_scores.summary)
    print("Instance Results:\n", results.instance_scores.summary)