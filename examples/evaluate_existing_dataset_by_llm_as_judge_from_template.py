from unitxt import get_logger, get_settings, load_dataset
from unitxt.api import evaluate
from unitxt.inference import (
    CrossProviderInferenceEngine,
)

logger = get_logger()
settings = get_settings()

with settings.context(allow_unverified_code=True):
    # Use the HF load_dataset API, to load the squad QA dataset using the standard template in the catalog.
    # We set loader_limit to 20 to reduce download time.
    dataset = load_dataset(
        card="cards.squad",
        template="templates.qa.with_context.simple",
        format="formats.chat_api",
        metrics=[
            "metrics.llm_as_judge.rating.llama_3_70b_instruct.generic_single_turn"
        ],
        loader_limit=20,
        max_test_instances=20,
        split="test",
    )

    # Infer a model to get predictions.
    model = CrossProviderInferenceEngine(
        model="llama-3-2-1b-instruct", provider="watsonx"
    )
    """
    We are using a CrossProviderInferenceEngine inference engine that supply api access to provider such as:
    watsonx, bam, openai, azure, aws and more.

    For the arguments these inference engines can receive, please refer to the classes documentation or read
    about the the open ai api arguments the CrossProviderInferenceEngine follows.
    """
    predictions = model(dataset)

    # Evaluate the predictions using the defined metric.
    results = evaluate(predictions=predictions, data=dataset)

    print("Global Results:")
    print(results.global_scores.summary)

    print("Instance Results:")
    print(results.instance_scores.summary)
