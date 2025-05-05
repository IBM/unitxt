import argparse
import logging
import os
import random
import socket
import threading
import time

import requests
from flask import Flask, jsonify, request
from werkzeug.serving import make_server

from ..inference import HFPipelineBasedInferenceEngine

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

class Server:
    def __init__(self, port: int):
        self.inference_engine = None
        self.inactivity_timeout = 600
        self.monitor_thread = threading.Thread(target=self.monitor_activity, daemon=True)
        self.last_request_time = time.time()
        self.shutdown_flag = False
        self.monitor_thread.start()
        self.port = port

    def update_last_request_time(self):
        self.last_request_time = time.time()

    def monitor_activity(self):
        while not self.shutdown_flag:
            time.sleep(5)
            if time.time() - self.last_request_time > self.inactivity_timeout:
                app.logger.info(f"No requests for {self.inactivity_timeout} seconds. Shutting down server...")
                try:
                    requests.post(f"http://localhost:{self.port}/shutdown", timeout=5)
                except Exception:
                    pass
            else:
                app.logger.info(
                    f"{int(self.inactivity_timeout - (time.time() - self.last_request_time))} till shutdown...")


    def shutdown_server(self):
        self.shutdown_flag = True
        app.logger.info("Server shutting down...")
        shutdown_func = request.environ.get("werkzeug.server.shutdown")
        if shutdown_func:
            shutdown_func()
        # Allow the shutdown process to complete, then force exit the program
        time.sleep(1)
        os._exit(0)  # This immediately stops the program

    def init_server(self, **kwargs):
        kwargs["use_cache"] =True
        self.inference_engine = HFPipelineBasedInferenceEngine(**kwargs)

    def infer(self, **kwargs):
        inputs = []
        return self.inference_engine(inputs)


@app.before_request
def update_activity():
    server.update_last_request_time()


@app.route("/shutdown", methods=["POST"])
def shutdown():
    app.logger.info("Received shutdown request")
    server.shutdown_server()
    return jsonify({"message": "Shutting down server..."}), 200


@app.route("/init_server", methods=["POST"])
def init_server():
    kwargs = request.get_json()
    server.init_server(**kwargs)
    return jsonify("Accepted")


@app.route("/<model>/v1/chat/completions", methods=["POST"])
@app.route("/<model_prefix>/<model>/v1/chat/completions", methods=["POST"])
def completions(model: str, model_prefix: str = "None"):
    if random.random() < 0:
        logging.error("Bad luck! Returning 500 with an error message.")
        app.logger.info("Server shutting down...")
        shutdown_func = request.environ.get("werkzeug.server.shutdown")
        if shutdown_func:
            shutdown_func()
        # Allow the shutdown process to complete, then force exit the program
        time.sleep(1)
        os._exit(0)  # This immediately stops the program
        return jsonify({"error": "Bad luck, something went wrong!"}), 500

    body = request.get_json()
    # validate that request parameters are equal to the model config. Print warnings if not.
    for k, v in body.items():
        if k == "messages":
            continue
        k = "model_name" if k == "model" else k
        attr = getattr(server.inference_engine, k, None)
        if attr is None:
            logging.warning(f"Warning: {k} is not an attribute in inference_engine")
        else:
            if attr != v:
                logging.warning(f"Warning: {k} value in boody({v}) is different from value in inference engine ({attr})")
    texts = [{"source": m[0]["content"]} for m in body["messages"]]
    predictions = server.inference_engine(texts)
    return jsonify({
        "choices": [{"message": {"role": "assistant","content": p}} for p in predictions],
    })


@app.route("/status", methods=["GET"])
def status():
    return "up", 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="unitxt inference worker server")
    parser.add_argument("--port", type=int, help="Port to run the server on", default=8080, required=False)
    args = parser.parse_args()
    server = Server(args.port)
    srv = make_server("0.0.0.0", args.port, app, threaded=True)
    # only here after bind succeeded
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    logging.info(f"server_ip={ip_address} server_port={args.port}")

    # this actually starts the Werkzeug loop (blocking)
    srv.serve_forever()


