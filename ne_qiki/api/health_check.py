from flask import Flask, jsonify
import torch

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    try:
        # Проверка доступности модели
        x = torch.randn(1, 16, 32)
        mask = torch.ones(1, 6).bool()
        return jsonify({"status": "ok", "model": "loaded"}), 200
    except Exception as e:
        return jsonify({"status": "error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
