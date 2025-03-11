from unitxt import add_to_catalog
from unitxt.inference import CrossProviderInferenceEngine
from unitxt.llm_as_judge_from_template import LLMAsJudge
from unitxt.random_utils import get_seed

model = "llama-3-3-70b-instruct"
format = "formats.chat_api"
template = "templates.safety.llamaguard"
task = "rating.single_turn"

inference_model = CrossProviderInferenceEngine(
    model=model, max_tokens=20, seed=get_seed(), temperature=1e-7
)

model_label = (
    model.replace("-", "_").replace(".", ",").lower() + "_cross_provider"
)

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
    "metrics.llm_as_judge.safety.llamaguard",
    overwrite=True,
)
