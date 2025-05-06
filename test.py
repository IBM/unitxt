import logging

from unitxt import evaluate, load_dataset
from unitxt.inference import (
    HFAutoModelInferenceEngine,
    HFPipelineBasedInferenceEngine,
    RITSInferenceEngine,
)

dataset = load_dataset(card="cards.pop_qa",
                       # format="formats.chat_api",
                       split="test")


# HFAutoModelInferenceEngine
# HFPipelineBasedInferenceEngine
model_name="meta-llama/Llama-3.1-8B-Instruct" # "meta-llama/Llama-3.2-1B-Instruct"
max_new_tokens=120
example_count = 8
pipeline_engine = HFPipelineBasedInferenceEngine(model_name=model_name, max_new_tokens=max_new_tokens, use_cache=False)
pipeline_engine_predictions = pipeline_engine.infer(dataset.select(range(example_count)))
auto_engine = HFAutoModelInferenceEngine(model_name=model_name, max_new_tokens=max_new_tokens, use_cache=False)
auto_engine_predictions = auto_engine.infer(dataset)



rits_engine = RITSInferenceEngine(model_name=model_name, max_new_tokens=max_new_tokens, use_cache=False)
rits_engine_predictions = auto_engine.infer(dataset.select(range(example_count)))

same_pred_count = sum(auto_engine_pred == pipeline_engine_pred for auto_engine_pred, pipeline_engine_pred
           in zip(auto_engine_predictions, pipeline_engine_predictions))
rits_pipeline_same_pred_count = sum(rits_engine_pred == pipeline_engine_pred for rits_engine_pred, pipeline_engine_pred
           in zip(rits_engine_predictions, pipeline_engine_predictions))
rits_auto_same_pred_count = sum(auto_engine_pred == rits_engine_pred for auto_engine_pred, rits_engine_pred
           in zip(auto_engine_predictions, rits_engine_predictions))
total_pred_count = len(pipeline_engine_predictions)

logging.info(f"{same_pred_count}/ {total_pred_count}")
logging.info(f"{rits_auto_same_pred_count}/ {total_pred_count}")

auto_engine_res = evaluate(auto_engine_predictions, dataset)
pipeline_engine_res = evaluate(pipeline_engine_predictions, dataset)

logging.info(auto_engine_res.global_scores.score)
logging.info(pipeline_engine_res.global_scores.score)

assert all(auto_engine_pred == pipeline_engine_pred for auto_engine_pred, pipeline_engine_pred
           in zip(auto_engine_predictions, pipeline_engine_predictions))
