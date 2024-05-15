from unitxt.inference import MockInferenceEngine
from unitxt.llm_as_judge import LLMAsJudge
from unitxt.test_utils.metrics import test_metric

model_id = "meta-llama/llama-3-8b-instruct"
task = "rating.single_turn"
format = "formats.llama3_chat"
template = "templates.response_assessment.rating.mt_bench_single_turn"

inference_model = MockInferenceEngine(model_name=model_id)
model_label = model_id.split("/")[1].replace("-", "_")
model_label = f"{model_label}_ibm_genai"
template_label = template.split(".")[-1]
metric_label = f"{model_label}_template_{template_label}"
metric = LLMAsJudge(
    inference_model=inference_model,
    task=task,
    template=template,
    format=format,
    main_score=metric_label,
)

predictions = ["[[10]]"] * 3
references = [["[[10]]"], ["[[10]]"], ["[[10]]"]]

instance_targets = [{metric_label: 1.0, "score_name": metric_label, "score": 1.0}] * 3

global_target = {
    metric_label: 1.0,
    "score": 1.0,
    "score_name": metric_label,
}

task_data = [
    {
        "input": "input",
        "type_of_input": "type",
        "output": "output",
        "type_of_output": "type",
        "source": "<SYS_PROMPT>input</SYS_PROMPT>",
        "metadata": {"template": "templates.generation.default"},
    }
] * 3

test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
    task_data=task_data,
)
