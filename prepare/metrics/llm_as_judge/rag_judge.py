from unitxt import add_to_catalog
from unitxt.inference import GenericInferenceEngine
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
    "answer_correctness": {"q_a_gt_loose": "judge_loose_match_no_context"},
    "answer_relevance": {"q_a": "judge_answer_relevance"},
}
metric_type_to_realization = {
    "faithfulness": "_verbal",
    "context_relevance": "_numeric",
    "correctness_holistic": "_numeric",
    "answer_correctness": "_numeric",
    "answer_relevance": "_numeric",
}

generic_engine_label = "generic_inference_engine"
inference_models = {
    "llama_3_1_70b_instruct_wml": "engines.classification.llama_3_1_70b_instruct_wml",
    generic_engine_label: GenericInferenceEngine(),
}


def get_prediction_field(metric_type):
    return "contexts" if metric_type == "context_relevance" else "answer"


for metric_type, template_dict in metric_type_to_template_dict.items():
    for template_short_name, template_name in template_dict.items():
        judge_task_name = f"tasks.rag_eval.{metric_type}.binary"
        for logprobs_label in [
            "",
            "_logprobs",
            metric_type_to_realization[metric_type],
        ]:
            use_logprobs = logprobs_label == "_logprobs"
            template = (
                f"templates.rag_eval.{metric_type}.{template_name}{logprobs_label}"
            )

            for inf_label, inference_model in inference_models.items():
                if (
                    use_logprobs and inf_label == generic_engine_label
                ):  # engine GenericInferenceEngine does not support logprobs
                    continue

                metric_label = f"{metric_type}_{template_short_name}{logprobs_label}"
                metric = TaskBasedLLMasJudge(
                    inference_model=inference_model,
                    template=template,
                    task=judge_task_name,
                    format=None,
                    main_score=metric_label,
                    prediction_field=get_prediction_field(metric_type),
                    infer_log_probs=use_logprobs,
                )

                new_catalog_name = f"metrics.rag.{metric_type}.{inf_label}_{template_short_name}{logprobs_label}"

                add_to_catalog(
                    metric,
                    new_catalog_name,
                    overwrite=True,
                )

                if logprobs_label in ["_logprobs", ""]:
                    metric = TaskBasedLLMasJudge(
                        inference_model=inference_model,
                        template=template,
                        task=judge_task_name,
                        format=None,
                        main_score=metric_label,
                        prediction_field=get_prediction_field(metric_type),
                        infer_log_probs=use_logprobs,
                        __deprecated_msg__=f"This metric should be replaced with {new_catalog_name}",
                    )
                    # for backwards compatibility: keep also legacy path to metrics
                    add_to_catalog(
                        metric,
                        f"metrics.llm_as_judge.binary.{inf_label}_{metric_label}",
                        overwrite=True,
                    )


# now add new metrics under unitxt rag tasks
metric_type_to_template_v2 = {
    "faithfulness": "judge_with_question_simplified",
    "context_relevance": "judge_context_relevance_ares",
    "answer_correctness": "judge_loose_match_no_context",
    "answer_relevance": "judge_answer_relevance",
}

inference_models_v2 = {
    "llama_3_3_70b_instruct_watsonx": "engines.classification.llama_3_3_70b_instruct_watsonx",
    "llama_3_3_70b_instruct_rits": "engines.classification.llama_3_3_70b_instruct_rits",
    "gpt_4o_azure": "engines.classification.gpt_4o_2024_08_06_azure_openai",
    generic_engine_label: GenericInferenceEngine(),
}

for metric_type, template_name in metric_type_to_template_v2.items():
    judge_task_name = f"tasks.rag_eval.{metric_type}.binary"
    realization_sufffix = metric_type_to_realization[metric_type]
    template = f"templates.rag_eval.{metric_type}.{template_name}{realization_sufffix}"
    for inf_label, inference_model in inference_models_v2.items():
        for rag_unitxt_task in ["autorag", "response_generation", "end_to_end"]:
            if (
                rag_unitxt_task == "response_generation"
                and metric_type == "context_relevance"
            ):
                continue

            judge_to_generator_fields_mapping = (
                {}
                if rag_unitxt_task == "autorag"
                else {"ground_truths": "reference_answers"}
            )

            new_catalog_name = (
                f"metrics.rag.{rag_unitxt_task}.{metric_type}.{inf_label}_judge"
            )
            metric = TaskBasedLLMasJudge(
                inference_model=inference_model,
                template=template,
                task=judge_task_name,
                format=None,
                main_score=f"{metric_type}_judge",
                prediction_field=get_prediction_field(metric_type),
                infer_log_probs=False,
                judge_to_generator_fields_mapping=judge_to_generator_fields_mapping,
            )
            add_to_catalog(
                metric,
                new_catalog_name,
                overwrite=True,
            )
