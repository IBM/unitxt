
from flask import Flask, jsonify

app = Flask(__name__)


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
