from unitxt import add_to_catalog, get_from_catalog
from unitxt.inference import (
    IbmGenAiInferenceEngine,
    OpenAiInferenceEngine,
    WMLInferenceEngine,
)
from unitxt.llm_as_judge import (
    TaskBasedLLMasJudge,
)
from unitxt.random_utils import get_seed

metric_type_to_template = {
    "faithfulness": "judge_with_question_simplified_logprobs",
    "context_relevance": "judge_context_relevance_ares_logprobs",
    "correctness_holistic": "judge_correctness_simple_logprobs",
    "answer_correctness": "judge_loose_match_no_context_logprobs",
    "answer_relevance": "judge_answer_relevance_logprobs",
}


def get_prediction_field(metric_type):
    return None if metric_type == "context_relevance" else "answer"


def get_inference_engine(model_name, framework_name):
    if framework_name == "wml":
        return WMLInferenceEngine(
            model_name=model_name,
            max_new_tokens=5,
            random_seed=get_seed(),
            decoding_method="greedy",
        )
    if framework_name == "bam":
        return IbmGenAiInferenceEngine(
            model_name=model_name,
            max_new_tokens=5,
            random_seed=get_seed(),
            decoding_method="greedy",
        )
    if framework_name == "openai":
        return OpenAiInferenceEngine(
            model_name=model_name, logprobs=True, max_tokens=5, temperature=0.0
        )

    raise ValueError("Unsupported framework name " + framework_name)


model_names_to_infer_framework = {
    "meta-llama/llama-3-1-70b-instruct": ["wml"],
    "meta-llama/llama-3-70b-instruct": ["bam"],
    "gpt-4-turbo": ["openai"],
    "mistralai/mixtral-8x7b-instruct-v01": ["wml", "bam"],
    "meta-llama/llama-3-1-405b-instruct-fp8": ["bam"],
}

for judge_model_name, infer_frameworks in model_names_to_infer_framework.items():
    template_format = (
        "formats.llama3_instruct" if "llama" in judge_model_name else "formats.empty"
    )
    for (
        metric_type,
        template_name,
    ) in metric_type_to_template.items():
        task_name = f"tasks.rag_eval.{metric_type}.binary"
        task_dict = get_from_catalog(task_name).input_fields
        for infer_framework in infer_frameworks:
            inference_engine = get_inference_engine(judge_model_name, infer_framework)

            model_label = (
                judge_model_name.split("/")[-1]
                .replace("-", "_")
                .replace(".", ",")
                .lower()
            )
            model_label = f"{model_label}_{infer_framework}"
            template_label = f'{metric_type}_{template_name.split(".")[-1]}'
            metric_label = f"{model_label}_template_{template_label}"
            metric = TaskBasedLLMasJudge(
                inference_model=inference_engine,
                template=f"templates.rag_eval.{metric_type}.{template_name}",
                task=task_name,
                format=template_format,
                main_score=metric_label,
                prediction_field=get_prediction_field(metric_type),
            )

            add_to_catalog(
                metric,
                f"metrics.llm_as_judge.binary.{metric_label}",
                overwrite=True,
            )
