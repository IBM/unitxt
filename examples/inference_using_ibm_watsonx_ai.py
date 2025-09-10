from unitxt.api import load_dataset
from unitxt.inference import WMLInferenceEngineGeneration

if __name__ == "__main__":
    # Set required env variables using your WML credentials:
    # os.environ["WML_URL"] = ""
    # os.environ["WML_PROJECT_ID"] = ""
    # os.environ["WML_APIKEY"] = ""

    # Preparing WML inference engine:
    model_name = "google/flan-t5-xl"
    wml_inference = WMLInferenceEngineGeneration(
        model_name=model_name,
        data_classification_policy=["public"],
        random_seed=111,
        min_new_tokens=16,
        max_new_tokens=128,
        top_p=0.5,
        top_k=1,
        repetition_penalty=1.5,
        decoding_method="greedy",
    )

    # Loading dataset:
    dataset = load_dataset(
        card="cards.go_emotions.simplified",
        template="templates.classification.multi_label.empty",
        loader_limit=3,
    )
    test_data = dataset["test"]

    # Performing inference:
    predictions = wml_inference(test_data)
    for instance, prediction in zip(test_data, predictions):
        print("*" * 10)
        print("source:", instance["source"])
        print("prediction:", prediction)
