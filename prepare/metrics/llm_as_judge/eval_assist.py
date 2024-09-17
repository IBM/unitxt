import json

from unitxt import add_to_local_catalog
from unitxt.inference import HFPipelineBasedInferenceEngine
from unitxt.metrics import (
    EvalAssistLLMAsJudge
)
from unitxt.templates import InputOutputTemplate, TemplatesDict

config_filepath = "prepare/metrics/llm_as_judge/eval_assist.json"

with open(config_filepath) as file:
    config = json.load(file)
print("config is ", config)

# judge_correctness_template = InputOutputTemplate(
#     instruction="Please act as an impartial judge and evaluate if the assistant's answer is correct."
#     ' Answer "[[10]]" if the answer is accurate, and "[[0]]" if the answer is wrong. '
#     'Please use the exact format of the verdict as "[[rate]]". '
#     "You can explain your answer after the verdict"
#     ".\n\n",
#     input_format="[User's input]\n{question}\n" "[Assistant's Answer]\n{answer}\n",
#     output_format="[[{rating}]]",
#     postprocessors=[
#         r"processors.extract_mt_bench_rating_judgment",
#     ],
# )

# platform = "hf"
# model_name = "google/flan-t5-base"
# inference_model = HFPipelineBasedInferenceEngine(
#     model_name=model_name, max_new_tokens=256
# )

eval_assist_metric = EvalAssistLLMAsJudge()#inference_model=inference_model, template=judge_correctness_template)

add_to_local_catalog(
    eval_assist_metric,
    "metrics.llm_as_judge.eval_assist.direct",
    overwrite=True
)