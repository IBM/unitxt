from unitxt import evaluate, load_dataset, settings
from unitxt.inference import CrossProviderInferenceEngine
from unitxt.text_utils import print_dict

with settings.context(
    disable_hf_datasets_cache=False,
    allow_unverified_code=True,
):
    test_dataset = load_dataset(
        "card=cards.text2sql.bird"
        ",template=templates.text2sql.you_are_given_with_hint_with_sql_prefix"
        ",loader_limit=100",
        split="validation",
    )

# Infer
inference_model = CrossProviderInferenceEngine(
    model="llama-3-70b-instruct",
    max_tokens=256,
)

predictions = inference_model.infer(test_dataset)
evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

# for item in evaluated_dataset:
#     print("\n\n------prediction--------")
#     print(item["prediction"])
#     print("------processed_prediction--------")
#     print(item["processed_prediction"])
#     print(evaluated_dataset[0]["score"]["instance"]["score"])

print_dict(
    evaluated_dataset[0],
    keys_to_print=[
        "source",
        "prediction",
        "subset",
    ],
)
print_dict(
    evaluated_dataset[0]["score"]["global"],
)

# with llama-3-70b-instruct
# num_of_instances (int):
#     1534
# execution_accuracy (float):
#     0.482

# like GPT4 (rank 40 in the benchmark https://bird-bench.github.io/)

# from transformers import AutoModelForCausalLM, AutoTokenizer

# DEBUG_NUM_EXAMPLES = 2
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token
# test_dataset = test_dataset.select(range(DEBUG_NUM_EXAMPLES))
# predictions = tokenizer.batch_decode(
#     model.generate(
#         **tokenizer.batch_encode_plus(
#             test_dataset["source"], return_tensors="pt", padding=True
#         ),
#         max_length=2048,
#     ),
#     skip_special_tokens=True,
#     clean_up_tokenization_spaces=True,
# )
