from unitxt import get_logger, produce
from unitxt.inference import GenericInferenceEngine

if __name__ == "__main__":
    generic_engine = GenericInferenceEngine(
        default="engines.ibm_gen_ai.llama_3_8b_instruct"
    )
    recipe = "card=cards.almost_evil,template=templates.qa.open.simple,demos_pool_size=0,num_demos=0"
    instances = [
        {"question": "How many days there are in a week", "answers": ["7"]},
        {
            "question": "If a ate an apple in the morning, and one in the evening, how many apples did I eat?",
            "answers": ["2"],
        },
    ]
    dataset = produce(instances, recipe)

    predictions = generic_engine.infer(dataset)

    get_logger().info(predictions)
