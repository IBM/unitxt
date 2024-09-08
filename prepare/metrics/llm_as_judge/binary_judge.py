from unitxt import add_to_catalog, get_from_catalog
from unitxt.inference import (
    IbmGenAiInferenceEngine,
    OpenAiInferenceEngine,
    WMLInferenceEngine,
)
from unitxt.llm_as_judge import (
    LLMAsJudge,
)
from unitxt.metrics import MetricPipeline
from unitxt.operators import Copy
from unitxt.random_utils import get_seed
from unitxt.stream import MultiStream

from examples.evaluate_rag_judge import test_examples

metric_type_to_template = {
    "faithfulness": "judge_with_question_simplified_logprobs",
    "context_relevance": "judge_context_relevance_ares_logprobs",
    "correctness_holistic": "judge_correctness_simple_logprobs",
    "answer_correctness": "judge_loose_match_no_context_logprobs",
    "answer_relevance": "judge_answer_relevance_logprobs",
}


def get_inference_engine(model_name, framework_name):
    if framework_name == "wml":
        return WMLInferenceEngine(
            model_name=model_name, max_new_tokens=5, random_seed=get_seed()
        )
    if framework_name == "bam":
        return IbmGenAiInferenceEngine(
            model_name=model_name, max_new_tokens=5, random_seed=get_seed()
        )
    if framework_name == "openai":
        return OpenAiInferenceEngine(model_name=model_name, logprobs=True, max_tokens=5)

    raise ValueError("Unsupported framework name " + framework_name)


model_names_to_infer_framework = {
    "meta-llama/llama-3-1-70b-instruct": "wml",
    "meta-llama/llama-3-70b-instruct": "bam",
    # "gpt-4-turbo": "openai",
    "mistralai/mixtral-8x7b-instruct-v01": "wml",
    # "meta-llama/llama-3-1-405b-instruct-fp8": "bam",
}

for judge_model_name, infer_framework in model_names_to_infer_framework.items():
    template_format = (
        "formats.llama3_instruct" if "llama" in judge_model_name else "formats.empty"
    )
    for (
        metric_type,
        template_name,
    ) in metric_type_to_template.items():
        task_name = f"tasks.rag_eval.{metric_type}.binary"
        task_dict = get_from_catalog(task_name).input_fields

        inference_engine = get_inference_engine(judge_model_name, infer_framework)

        model_label = (
            judge_model_name.split("/")[-1].replace("-", "_").replace(".", ",").lower()
        )
        model_label = f"{model_label}_{infer_framework}"
        template_label = f'{metric_type}_{template_name.split(".")[-1]}'
        metric_label = f"{model_label}_template_{template_label}"
        metric = LLMAsJudge(
            inference_model=inference_engine,
            template=f"templates.rag_eval.{metric_type}.{template_name}",
            task=task_name,
            format=template_format,
            main_score=metric_label,
            prediction_type=str,
            strip_system_prompt_and_format_from_inputs=False,
        )

        metric_pipeline = MetricPipeline(
            main_score=metric_label,
            metric=metric,
            preprocess_steps=[
                Copy(
                    field_to_field={
                        "data_classification_policy": "task_data/data_classification_policy"
                    },
                    not_exist_ok=True,
                    get_default=["public"],
                ),
                Copy(
                    field_to_field={
                        "prediction": "prediction",
                    },
                    not_exist_ok=True,
                    get_default="0.0",
                ),
                Copy(
                    field_to_field={
                        "references": "references",
                        "choices": "choices",
                    },
                    not_exist_ok=True,
                    get_default=["0.0"],
                ),
                Copy(
                    field_to_field={
                        field: f"task_data/{field}" for field in task_dict.keys()
                    },
                ),
            ],
        )

        multi_stream = MultiStream.from_iterables({"test": test_examples}, copying=True)
        output_stream = list(metric_pipeline(multi_stream)["test"])
        metric_outputs = [
            ex["score"]["instance"][metric_label] for ex in list(output_stream)
        ]
        # print(metric_label)
        # print(metric_outputs)

        add_to_catalog(
            metric_pipeline,
            f"metrics.llm_as_judge.binary.{metric_label}",
            overwrite=True,
        )
