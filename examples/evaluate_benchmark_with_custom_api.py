import unitxt
from unitxt import evaluate, get_from_catalog, load_dataset
from unitxt.text_utils import print_dict

with unitxt.settings.context(
    default_inference_api="watsonx",  # option a to define your home api
    default_format="formats.chat_api",
    disable_hf_datasets_cache=False,
):
    data = load_dataset("benchmarks.glue[max_samples_per_subset=5]", split="test")

    model = get_from_catalog(
        "engines.model.llama_3_8b_instruct[api=watsonx,top_k=1]"
    )  # option b to define your home api

    predictions = model.infer(data)

    evaluated_dataset = evaluate(predictions=predictions, data=data)

    print_dict(
        evaluated_dataset[0],
        keys_to_print=[
            "source",
            "prediction",
            "subset",
        ],
    )
    print_dict(
        evaluated_dataset[0]["score"]["subsets"],
    )
