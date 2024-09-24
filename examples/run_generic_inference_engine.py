from unitxt import get_logger, produce  # Import necessary functions from unitxt
from unitxt.inference import GenericInferenceEngine  # Import the inference engine class

if __name__ == "__main__":
    # Create an instance of the GenericInferenceEngine with a default engine.
    # This means if no engine is specified during inference, it will default to this one.
    generic_engine_with_default = GenericInferenceEngine(
        default="engines.ibm_gen_ai.llama_3_70b_instruct"
    )

    # Define the recipe for data processing and model selection.
    # - card: Specifies the underlying data (from cards.almost_evil).
    # - template: Selects the specific template within the card (from templates.qa.open.simple).
    # - demos_pool_size and num_demos: Control the number of demonstration examples used (set to 0 here).
    recipe = "card=cards.almost_evil,template=templates.qa.open.simple,demos_pool_size=0,num_demos=0"

    # Create a list of instances (data points) for inference.
    # Each instance has a "question" and its corresponding "answers".
    instances = [
        {
            "question": "How many days there are in a week, answer only with numerals",
            "answers": ["7"],
        },
        {
            "question": "If a ate an apple in the morning, and one in the evening, what is the number of apples I have eaten?, answer only with numerals",
            "answers": ["2"],
        },
    ]

    # Process the instances using the defined recipe.
    # This likely formats the data according to the chosen card and template.
    dataset = produce(instances, recipe)

    # Perform inference on the processed dataset using the engine with the default model.
    predictions = generic_engine_with_default.infer(dataset)
    get_logger().info(predictions)  # Log the predictions

    # The following code block demonstrates how to use the GenericInferenceEngine without specifying a
    # default engine. It expects the engine to be defined in the UNITXT_INFERENCE_ENGINE environment variable.
    try:
        # Attempt to create an instance without a default engine.
        generic_engine_without_default = GenericInferenceEngine()

        # Perform inference (will use the engine specified in the environment variable).
        predictions = generic_engine_without_default.infer(dataset)
        get_logger().info(predictions)  # Log the predictions
    except:
        # Handle the case where the environment variable is not set.
        get_logger().error(
            "GenericInferenceEngine could not be initialized without a default since "
            "UNITXT_INFERENCE_ENGINE environmental variable is not set."
        )
