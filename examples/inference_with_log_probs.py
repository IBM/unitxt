from typing import List

from unitxt.api import load_dataset
from unitxt.inference import (
    HFAutoModelInferenceEngine,
    TextGenerationInferenceOutput,
    WMLInferenceEngineGeneration,
)

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
        min_new_tokens=3,
        max_new_tokens=10,
        top_p=0.5,
        top_k=1,
        repetition_penalty=1.5,
        decoding_method="greedy",
    )

    hf_model = HFAutoModelInferenceEngine(
        model_name="google/flan-t5-small", max_new_tokens=10
    )

    # Loading dataset:
    dataset = load_dataset(
        card="cards.go_emotions.simplified",
        template="templates.classification.multi_label.default",
        loader_limit=1,
    )
    test_data = dataset["test"]

    for model in [wml_inference, hf_model]:
        # Performing inference:
        predictions: List[TextGenerationInferenceOutput] = model.infer_log_probs(
            test_data, return_meta_data=True
        )
        for instance, prediction in zip(test_data, predictions):
            print("*" * 80)
            print("model:", model.__class__)
            print("source:", instance["source"])
            print("generated_text:", prediction.generated_text)
            print(
                "predicated top tokens:",
                [token["text"] for token in prediction.prediction],
            )
            print("*" * 80)
