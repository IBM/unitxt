
import logging
import os
import random
import sys
import time

from flask import Flask, jsonify, request
from unitxt.inference import HFPipelineBasedInferenceEngine

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)


class Server:
    def __init__(self):
        self.inference_engine = None

    def init_server(self, **kwargs):
        kwargs["use_cache"] =True
        self.inference_engine = HFPipelineBasedInferenceEngine(**kwargs)

    def infer(self, **kwargs):
        inputs = []
        return self.inference_engine(inputs)


server = Server()


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
    app.run(host="0.0.0.0", port=sys.argv[1], debug=True)
