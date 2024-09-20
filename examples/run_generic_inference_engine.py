from unitxt import get_logger, produce
from unitxt.inference import GenericInferenceEngine

if __name__ == "__main__":
    generic_engine_with_default = GenericInferenceEngine(
        default="engines.ibm_gen_ai.llama_3_70b_instruct"
    )
    recipe = "card=cards.almost_evil,template=templates.qa.open.simple,demos_pool_size=0,num_demos=0"
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
    dataset = produce(instances, recipe)

    # now, trying without a default, make sure you have something like
    # export UNITXT_INFERENCE_ENGINE="engines.ibm_gen_ai.llama_3_70b_instruct"
    # in your ~/.bashrc
    predictions = generic_engine_with_default.infer(dataset)
    get_logger().info(predictions)

    try:
        generic_engine_without_default = GenericInferenceEngine()
        predictions = generic_engine_without_default.infer(dataset)
        get_logger().info(predictions)
    except:
        get_logger().error(
            "GenericInferenceEngine could not be initialized without a default since "
            "UNITXT_INFERENCE_ENGINE environmental variable is not set."
        )
