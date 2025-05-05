import hashlib
import json
import logging
import os
import subprocess
import time

import joblib
import requests
import unitxt
from unitxt import load_dataset
from unitxt.inference import MultiServersInferenceEngine
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


def run_worker_in_a_port(port):
    kill_process_on_port(port)
    process = subprocess.Popen(
        ["/Users/eladv/miniforge3/envs/unitxt/bin/python", "/Users/eladv/unitxt/ccc_worker_server.py", f"{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True)
    logger.info(f"Started worker on port {port} with PID {process.pid}")
    return process

def kill_process_on_port(port):
    try:
        output = subprocess.check_output(f"lsof -ti:{port}", shell=True).decode().strip()
        if output:
            logger.info(f"Killing process {output} on port {port}...")
            subprocess.run(f"kill -9 {output}", shell=True)
    except subprocess.CalledProcessError:
        pass


def is_up(server_url):
    try:
        response = requests.get(f"{server_url}/status", timeout=5)
        return response.text.strip().lower() == "up"
    except requests.RequestException:
        return False

def set_up_worker_servers(servers):
    ports = [5000, 5001, 5002, 5003, 5004]
    processes = [(port, run_worker_in_a_port(port)) for port in ports]
    while len(processes) > 0:
        for port, process in processes:
            # Check if the process is still running
            if process.poll() is not None:  # If poll() returns None, the process is still running
                stdout, stderr = process.communicate()  # Get the output and errors
                logger.error(f"Process on port {port} has stopped!")
                logger.error(f"STDOUT:\n{stdout}")
                logger.error(f"STDERR:\n{stderr}")
                raise RuntimeError(f"Failed to Start server on port: {port}")
            if is_up(f"http://localhost:{port}"):
                processes.remove((port, process))
        time.sleep(0.3)
        logger.info(f"The following servers still need to start: {[p[0] for p in processes]}")


if __name__ == "__main__":
    #ports = [5000,5001,5002,5003,5004]
    #servers = [f"http://localhost:{port}" for port in ports]
    hosts = ["cccxc425","cccxc417"]# ,'cccxc436']
    servers = [f"http://{server}.pok.ibm.com:5000" for server in hosts]
    #set_up_worker_servers(servers)
    set_verbosity("debug")
    unitxt.settings.allow_unverified_code = True
    dataset = load_dataset_cached(card="cards.openbook_qa",
                                  split="test")

    dataset = dataset.select(range(100))

    inference_model = MultiServersInferenceEngine(
        model_name="google/flan-t5-small",
        temperature=0.202,
        max_new_tokens=256,
        use_cache=True,
        cache_batch_size=5,
        workers_url=servers
    )

    start_time = time.time()
    predictions = inference_model.infer(dataset)
    end_time = time.time()

    logger.info(f"predictions contains {predictions.count(None)} Nones")
    for p in predictions:
        logger.info(f"prediction: {p}")
