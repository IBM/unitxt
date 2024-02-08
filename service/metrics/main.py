import datetime
import logging
import threading
import time
from logging import Formatter, StreamHandler, getLevelName, getLogger
from typing import cast

import torch
import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import HTTPException
from starlette.responses import JSONResponse
from tokens import verify_token

from src.unitxt.metric_utils import MetricRequest, MetricResponse

"""
This module defines an http server that wraps unitxt metrics.
It accepts requests detailing which metric to run, and what is the data to run on.
The requests are handled by running them one by one locally, potentially on a GPU.
"""

# init the FastAPI app object
app = FastAPI(version="0.0.1", title="Unitxt Metrics Service")


def init_logger():
    log = getLogger()
    log.setLevel(getLevelName("INFO"))
    log_formatter = Formatter(
        "%(asctime)s [%(levelname)s] %(filename)s %(lineno)d: %(message)s [%(threadName)s]"
    )

    console_handler = StreamHandler()
    console_handler.setFormatter(log_formatter)
    log.handlers = []
    log.addHandler(console_handler)


init_logger()


# for sanity check
@app.get("/", include_in_schema=False)
def read_root():
    return {"Hello": "Unitxt Metrics"}


# for k8s health checks
@app.get("/health", include_in_schema=False)
def health():
    return "OK"


# A lock to make sure single use of the GPU
compute_lock = threading.Lock()


# for computing a metric
@app.post("/compute/{metric}", response_model=MetricResponse)
def compute(metric: str, request: MetricRequest, token: dict = Depends(verify_token)):
    # imports are here, so the service could start even if unitxt is not installed.
    # This is useful for testing, it enabled running health checks and sanity checks, without unitxt.
    from unitxt.artifact import Artifact
    from unitxt.operator import MultiStreamOperator
    from unitxt.operators import ArtifactFetcherMixin
    from unitxt.stream import MultiStream

    t0 = time.perf_counter()
    try:
        logging.info(f"Request from [{token['sub']}]")
        logging.info(f"Computing metric '{metric}'.")
        logging.info(
            f"MetricRequest contains {len(request.instance_inputs)} input instances"
        )

        start_time = datetime.datetime.now()
        # Only allow single use of the GPU, other requests wait on this lock, till
        # current computation is done.
        with compute_lock:
            logging.info("Acquired compute_lock, starting computation .. ")
            start_infer_time = datetime.datetime.now()
            # obtain the metric to compute
            metric_artifact: Artifact = ArtifactFetcherMixin.get_artifact(metric)
            metric_artifact: MultiStreamOperator = cast(
                MultiStreamOperator, metric_artifact
            )

            # prepare the input stream
            multi_stream: MultiStream = MultiStream.from_iterables(
                {"test": request.model_dump()["instance_inputs"]}, copying=True
            )

            # apply the metric and obtain the results
            metric_results = list(metric_artifact(multi_stream)["test"])

        infer_time = datetime.datetime.now() - start_infer_time
        wait_time = start_infer_time - start_time
        logging.info(
            f"Computed {len(metric_results)} metric '{metric}' results, "
            f"took: {infer_time!s}, waited: {wait_time!s}')"
        )

        metric_response = {
            "instances_scores": [
                metric_result["score"]["instance"] for metric_result in metric_results
            ],
            "global_score": metric_results[0]["score"]["global"],
        }
        return MetricResponse.model_validate(metric_response)
    finally:
        t1 = time.perf_counter()
        logging.info(f"Request for metric '{metric}' handled in [{t1 - t0:.2f}] secs.")


# wrapper for HTTP exceptions that we throw
@app.exception_handler(HTTPException)
async def unicorn_http_exception_handler(_request: Request, exc: HTTPException):
    logging.exception("HTTP Exception raised")
    return JSONResponse(
        status_code=exc.status_code,
        headers=exc.headers,
        content={"message": exc.detail},
    )


# wrapper for unexpected exceptions
@app.exception_handler(Exception)
async def unicorn_exception_handler(_request: Request, exc: Exception):
    logging.exception(f"Unexpected exception raised: {type(exc).__name__}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error"},
    )


def print_gpus_status():
    if torch.cuda.is_available():
        logging.info("Using CUDA")
        logging.info(f"CUDNN VERSION: {torch.backends.cudnn.version()}")
        gpu_id = torch.cuda.current_device()
        logging.info(
            f"There are {torch.cuda.device_count()} GPUs available, using GPU {gpu_id}, name: {torch.cuda.get_device_name(gpu_id)}"
        )
        logging.info(
            f"CUDA Device Total Memory [GB]: {torch.cuda.get_device_properties(0).total_memory / 1e9}"
        )
    else:
        logging.info("There are NO GPUs available.")


def start_metrics_http_service():
    print_gpus_status()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_config=None)


if __name__ == "__main__":
    start_metrics_http_service()
