from unitxt import add_to_catalog
from unitxt.llm_as_judge import (
    TaskBasedLLMasJudge,
)

metric_type_to_template_dict = {
    "faithfulness": {
        "q_c_a_logprobs": "judge_with_question_simplified_logprobs",
        "c_a_logprobs": "judge_no_question_simplified_logprobs",
    },
    "context_relevance": {"q_c_ares_logprobs": "judge_context_relevance_ares_logprobs"},
    "correctness_holistic": {"q_c_a_logprobs": "judge_correctness_simple_logprobs"},
    "answer_correctness": {
        "q_a_gt_loose_logprobs": "judge_loose_match_no_context_logprobs",
        "q_a_gt_strict_logprobs": "judge_simplified_format",
    },
    "answer_relevance": {"q_a_logprobs": "judge_answer_relevance_logprobs"},
}


def get_prediction_field(metric_type):
    return None if metric_type == "context_relevance" else "answer"


for metric_type, template_dict in metric_type_to_template_dict.items():
    for template_short_name, template_name in template_dict.items():
        task_name = f"tasks.rag_eval.{metric_type}.binary"

        metric_label = f"{metric_type}_{template_short_name}"
        metric = TaskBasedLLMasJudge(
            inference_model="engines.classification.llama_3_1_70b_instruct_wml",
            template=f"templates.rag_eval.{metric_type}.{template_name}",
            task=task_name,
            format="formats.empty",
            main_score=metric_label,
            prediction_field=get_prediction_field(metric_type),
        )

        add_to_catalog(
            metric,
            f"metrics.llm_as_judge.binary.{metric_label}",
            overwrite=True,
        )
