from unitxt import add_to_catalog
from unitxt.inference import (
    CrossProviderInferenceEngine,
    GenericInferenceEngine,
    WMLInferenceEngine,
)
from unitxt.llm_as_judge import LLMAsJudge

model_list = [
    "meta-llama/llama-3-8b-instruct",
    "meta-llama/llama-3-70b-instruct",
    "meta-llama/llama-3-3-70b-instruct",
]
format = "formats.llama3_instruct"
templates = [
    "templates.response_assessment.pairwise_comparative_rating.arena_hard",
    "templates.response_assessment.pairwise_comparative_rating.arena_hard_with_shuffling",
]

inference_engines = [
    ("ibm_wml", WMLInferenceEngine),
    ("generic_engine", GenericInferenceEngine),
]


for template in templates:
    task = "pairwise_comparative_rating.single_turn"

    for model_id in model_list:
        for inference_engine_name, inference_engine in inference_engines:
            if (
                inference_engine_name == "ibm_wml"
                and model_id == "meta-llama/llama-3-8b-instruct"
            ):
                continue  # currently not supported

            # if inference engine is generic, these configurations will be defined when it is saved to the catalog
            if inference_engine_name != "generic_engine":
                inference_model = inference_engine(
                    model_name=model_id, max_new_tokens=2048, random_seed=42
                )
            else:
                inference_model = inference_engine(
                    default="engines.ibm_gen_ai.llama_3_70b_instruct"
                )

            model_label = (
                model_id.split("/")[1].replace("-", "_").replace(".", ",").lower()
            )
            model_label = f"{model_label}_{inference_engine_name}"
            template_label = template.split(".")[-1]
            metric_label = f"{model_label}_template_{template_label}"
            metric = LLMAsJudge(
                inference_model=inference_model,
                template=template,
                task=task,
                format=format,
                main_score=metric_label,
            )

            add_to_catalog(
                metric,
                f"metrics.llm_as_judge.pairwise_comparative_rating.{model_label}_template_{template_label}",
                overwrite=True,
            )

add_to_catalog(
    LLMAsJudge(
        inference_model=CrossProviderInferenceEngine(
            model="llama-3-70b-instruct",
            max_tokens=30,
        ),
        template="templates.response_assessment.pairwise_comparative_rating.arena_hard",
        task="pairwise_comparative_rating.single_turn",
        format="formats.chat_api",
        main_score="llama_3_70b_instruct_template_arena_hard",
    ),
    "metrics.llm_as_judge.pairwise_comparative_rating.llama_3_70b_instruct.template_arena_hard",
    overwrite=True,
)

add_to_catalog(
    LLMAsJudge(
        inference_model=CrossProviderInferenceEngine(
            model="llama-3-8b-instruct",
            max_tokens=30,
        ),
        template="templates.response_assessment.pairwise_comparative_rating.arena_hard",
        task="pairwise_comparative_rating.single_turn",
        format="formats.chat_api",
        main_score="llama_3_8b_instruct_template_arena_hard",
    ),
    "metrics.llm_as_judge.pairwise_comparative_rating.llama_3_8b_instruct.template_arena_hard",
    overwrite=True,
)

add_to_catalog(
    LLMAsJudge(
        inference_model=CrossProviderInferenceEngine(
            model="llama-3-3-70b-instruct",
            max_tokens=2048,
        ),
        template="templates.response_assessment.pairwise_comparative_rating.arena_hard",
        task="pairwise_comparative_rating.single_turn",
        format="formats.chat_api",
        main_score="llama_3_70b_instruct_template_arena_hard",
    ),
    "metrics.llm_as_judge.pairwise_comparative_rating.llama_3_3_70b_instruct.template_arena_hard",
    overwrite=True,
)
