import logging

import torch
from unitxt import evaluate, load_dataset
from unitxt.inference import (
    HFAutoModelInferenceEngine,
    HFPipelineBasedInferenceEngine,
    RITSInferenceEngine,
)

dataset = load_dataset(card="cards.pop_qa",
                       format="formats.chat_api",
                       split="test")


# HFAutoModelInferenceEngine
# HFPipelineBasedInferenceEngine
model_name="meta-llama/Llama-3.1-70B-Instruct" # "meta-llama/Llama-3.2-1B-Instruct"
max_new_tokens=120
example_range = range(4)

logging.critical(torch.cuda.device_count())
auto_engine = HFAutoModelInferenceEngine(model_name=model_name, max_new_tokens=max_new_tokens, top_p=1, use_cache=False)
auto_engine_predictions = auto_engine.infer(dataset.select(example_range))


pipeline_engine = HFPipelineBasedInferenceEngine(model_name=model_name, max_new_tokens=max_new_tokens, top_p=1,
                                                use_cache=False)
pipeline_engine_predictions = pipeline_engine.infer(dataset.select(example_range))



rits_engine = RITSInferenceEngine(model_name=model_name, use_cache=False,max_tokens=120,top_p=1, temperature=0, seed=1)
rits_engine_predictions = rits_engine.infer(dataset.select(example_range))

rits2_engine = RITSInferenceEngine(model_name=model_name, use_cache=False,max_tokens=120,top_p=1, temperature=0, seed=1)
rits2_engine_predictions = rits2_engine.infer(dataset.select(example_range))


auto_pipeline_same_pred_count = sum(auto_engine_pred == pipeline_engine_pred for auto_engine_pred, pipeline_engine_pred
           in zip(auto_engine_predictions, pipeline_engine_predictions))

rits_auto_same_pred_count = sum(auto_engine_pred == rits_engine_pred for auto_engine_pred, rits_engine_pred
           in zip(auto_engine_predictions, rits_engine_predictions))

rits_pipeline_same_pred_count = sum(rits_engine_pred == pipeline_engine_pred for rits_engine_pred, pipeline_engine_pred
           in zip(rits_engine_predictions, pipeline_engine_predictions))
rits_rits2_same_pred_count = sum(rits_engine_pred == rits2_engine_pred for rits_engine_pred, rits2_engine_pred
           in zip(rits_engine_predictions, rits2_engine_predictions))

total_pred_count = len(pipeline_engine_predictions)

logging.critical(f"auto_pipeline_same_pred_count: {auto_pipeline_same_pred_count}/ {total_pred_count}")
logging.critical(f"rits_auto_same_pred_count: {rits_auto_same_pred_count}/ {total_pred_count}")
logging.critical(f"rits_pipeline_same_pred_count: {rits_pipeline_same_pred_count}/ {total_pred_count}")
logging.critical(f"rits_rits2_same_pred_count: {rits_rits2_same_pred_count}/ {total_pred_count}")
auto_engine_res = evaluate(auto_engine_predictions, dataset)
pipeline_engine_res = evaluate(pipeline_engine_predictions, dataset)
rits_engine_res = evaluate(rits_engine_predictions, dataset)
logging.critical(f"auto_engine_res: {auto_engine_res.global_scores.score}")
logging.critical(f"pipeline_engine_res: {pipeline_engine_res.global_scores.score}")
logging.critical(f"rits_engine_res: {rits_engine_res.global_scores.score}")
"""
assert all(auto_engine_pred == pipeline_engine_pred for auto_engine_pred, pipeline_engine_pred
           in zip(auto_engine_predictions, pipeline_engine_predictions))
"""
