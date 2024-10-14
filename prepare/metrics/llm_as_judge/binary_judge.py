from unitxt import add_to_catalog
from unitxt.llm_as_judge import (
    TaskBasedLLMasJudge,
)

metric_type_to_template_dict = {
    "faithfulness": {
        "q_c_a": "judge_with_question_simplified",
        "c_a": "judge_no_question_simplified",
    },
    "context_relevance": {"q_c_ares": "judge_context_relevance_ares"},
    "correctness_holistic": {"q_c_a": "judge_correctness_simple"},
    "answer_correctness": {
        "q_a_gt_loose": "judge_loose_match_no_context",
        "q_a_gt_strict": "judge_simplified_format",
    },
    "answer_relevance": {"q_a": "judge_answer_relevance"},
}


def get_prediction_field(metric_type):
    return None if metric_type == "context_relevance" else "answer"


for metric_type, template_dict in metric_type_to_template_dict.items():
    for template_short_name, template_name in template_dict.items():
        task_name = f"tasks.rag_eval.{metric_type}.binary"

        for use_logprobs in [True, False]:
            logprobs_label = "_logprobs" if use_logprobs else ""
            metric_label = f"{metric_type}_{template_short_name}{logprobs_label}"
            metric = TaskBasedLLMasJudge(
                inference_model="engines.classification.llama_3_1_70b_instruct_wml",
                template=f"templates.rag_eval.{metric_type}.{template_name}{logprobs_label}",
                task=task_name,
                format="formats.empty",
                main_score=metric_label,
                prediction_field=get_prediction_field(metric_type),
                infer_log_probs=use_logprobs,
            )

            add_to_catalog(
                metric,
                f"metrics.llm_as_judge.binary.{metric_label}",
                overwrite=True,
            )
