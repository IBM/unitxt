
from flask import Flask, jsonify, request
from unitxt.inference import HFPipelineBasedInferenceEngine

app = Flask(__name__)


class Server:
    def __init__(self):
        self.inference_engine = None

    def init_server(self, **kwargs):
        self.inference_engine = HFPipelineBasedInferenceEngine(**kwargs)

    def infer(self, **kwargs):
        inputs = []
        return self.inference_engine(inputs)


server = Server()


@app.route("/init_server", methods=["POST"])
def init_server():
    kwargs = request.get_json()
    server.init_server(**kwargs)


@app.route("/auth", methods=["POST"])
def auth():
    kwargs = request.get_json()
    server.init_server(**kwargs)
    return jsonify({})


@app.route("/<model>/v1/chat/completions", methods=["POST"])
def completions(model: str):
    #body = request.get_json()
    #print(f"Request body: {body}")

    return jsonify({
        "choices": [{"message": {
                    "role": "assistant",
                    "content": "I am great"
                },}],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
