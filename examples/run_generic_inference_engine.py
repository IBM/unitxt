from unitxt import get_logger, produce
from unitxt.inference import GenericInferenceEngine

if __name__ == "__main__":
    generic_engine = GenericInferenceEngine(
        default="engines.ibm_wml.llama_3_70b_instruct"
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

    predictions = generic_engine.infer(dataset)

    get_logger().info(predictions)
