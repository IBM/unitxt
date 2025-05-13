import hashlib
import json
import logging
import os
import time

import joblib
import unitxt
from unitxt import load_dataset
from unitxt.inference import CCCInferenceEngine
from unitxt.logging_utils import set_verbosity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


def get_cache_filename(cache_dir="cache", **kwargs):
    """Generate a unique filename for caching based on function arguments."""
    os.makedirs(cache_dir, exist_ok=True)
    hash_key = hashlib.md5(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()
    return os.path.join(cache_dir, f"dataset_{hash_key}.pkl")


def load_dataset_cached(**kwargs):
    """Load dataset with disk caching."""
    cache_file = get_cache_filename(**kwargs)

    if os.path.exists(cache_file):
        return joblib.load(cache_file)

    data = load_dataset(**kwargs)
    joblib.dump(data, cache_file)
    return data


if __name__ == "__main__":
    set_verbosity("debug")
    unitxt.settings.allow_unverified_code = True
    dataset = load_dataset_cached(card="cards.openbook_qa",
                                  split="test")

    dataset = dataset.select(range(200))

    inference_model = CCCInferenceEngine(
        model_name="google/flan-t5-small",
        temperature=0.202,
        max_new_tokens=256,
        use_cache=True,
        cache_batch_size=5,
        ccc_host="cccxl013.pok.ibm.com",
        ccc_user="ofirarviv",
        ccc_python="/dccstor/fme/users/ofir.arviv/miniforge3/envs/fme/bin/python",
        num_of_workers=3,
        ccc_queue = "nonstandard"
    )

    start_time = time.time()
    predictions = inference_model.infer(dataset)
    end_time = time.time()

    logger.info(f"predictions contains {predictions.count(None)} Nones")
    for p in predictions:
        logger.info(f"prediction: {p}")
