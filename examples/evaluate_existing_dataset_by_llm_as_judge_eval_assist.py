from unitxt import get_logger, get_settings, load_dataset
from unitxt.api import evaluate
from unitxt.inference import (
    CrossProviderInferenceEngine,
)
from unitxt.text_utils import print_dict

logger = get_logger()
settings = get_settings()

with settings.context(allow_unverified_code=True):
    # Use the HF load_dataset API, to load the squad QA dataset using the standard template in the catalog.
    # We set loader_limit to 20 to reduce download time.
    dataset = load_dataset(
        card="cards.squad",
        metrics=[
            "metrics.llm_as_judge.eval_assist.direct_assessment.rits.llama3_1_70b[context_field=context,criteria_or_criterias=metrics.llm_as_judge.eval_assist.direct_assessment.criterias.answer_relevance,score_prefix=answer_relevance_]"
        ],
        loader_limit=20,
        max_test_instances=20,
        split="test",
    )

    # Infer a model to get predictions.
    inference_model = CrossProviderInferenceEngine(
        model="llama-3-2-1b-instruct", provider="watsonx"
    )
    """
    We are using a CrossProviderInferenceEngine inference engine that supply api access to provider such as:
    watsonx, bam, openai, azure, aws and more.

    For the arguments these inference engines can receive, please refer to the classes documentation or read
    about the the open ai api arguments the CrossProviderInferenceEngine follows.
    """
    predictions = inference_model.infer(dataset)

    # Evaluate the predictions using the defined metric.
    evaluated_dataset = evaluate(predictions=predictions, data=dataset)

    print_dict(
        evaluated_dataset[0],
        keys_to_print=[
            "source",
            "prediction",
            "processed_prediction",
            "references",
            "score",
        ],
    )
