from unitxt.inference import CrossProviderInferenceEngine
from unitxt.text_utils import print_dict

if __name__ == "__main__":
    for provider in ["watsonx", "rits", "watsonx-sdk", "hf-local"]:
        print()
        print("------------------------------------------------ ")
        print("PROVIDER:", provider)
        model = CrossProviderInferenceEngine(
            model="granite-3-3-8b-instruct", provider=provider, temperature=0
        )

        # Loading dataset:
        test_data = [
            {
                "source": [{"content": "Hello, how are you?", "role": "user"}],
                "data_classification_policy": ["public"],
            }
        ]

        # Performing inference:
        predictions = model(test_data)
        for inp, prediction in zip(test_data, predictions):
            result = {**inp, "prediction": prediction}

            print_dict(result, keys_to_print=["source", "prediction"])
