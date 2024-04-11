import functools
import hashlib
import os
import pickle
import time

import evaluate
from datasets import load_dataset
from transformers import pipeline
from unitxt import get_logger

logger = get_logger()


def cache_func_in_file(func):
    """Decorator to cache function outputs to unique files based on parameters."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        args_hash = hashlib.sha256(pickle.dumps((args, kwargs))).hexdigest()
        cache_filename = f"{func.__name__}_{args_hash}_cache.pkl"

        if os.path.exists(cache_filename):
            logger.info(f"Find cache file {cache_filename}. Loading from cache...")
            with open(cache_filename, "rb") as cache_file:
                logger.info(f"{func.__name__} took {time.time() - start_time:.4f}")
                return pickle.load(cache_file)

        logger.info(
            f"Cache file for func {cache_filename} doesn't exists. Calculating..."
        )
        result = func(*args, **kwargs)

        logger.info(f"Saving result in a cache file: {func.__name__}")
        with open(cache_filename, "wb") as cache_file:
            pickle.dump(result, cache_file)
        logger.info(f"{func.__name__} took {time.time() - start_time:.4f}")
        return result

    return wrapper


def infer_llm(dataset, model):
    predictions = [
        output["generated_text"]
        for output in model(dataset["source"], max_new_tokens=30)
    ]
    return predictions


@cache_func_in_file
def create_predictions_for_ds_and_model(dataset, model):
    dataset = load_dataset("unitxt/data", dataset, split="train")
    model = pipeline(model=model)
    predictions = infer_llm(dataset, model)
    return predictions, dataset


def main():
    predictions, dataset = create_predictions_for_ds_and_model(
        dataset="card=cards.almost_evil,template=templates.qa.open.simple,"
                "metrics=[metrics.llm_as_judge.model_response_assessment.mt_bench_flan_t5],"
                "system_prompt=system_prompts.empty,max_train_instances=5",
        model="google/flan-t5-base",
    )
    metric = evaluate.load("unitxt/metric")
    scores = metric.compute(predictions=predictions, references=dataset)

    [print(item) for item in scores[0]["score"]["global"].items()]


if __name__ == "__main__":
    main()
