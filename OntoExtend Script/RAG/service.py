from flask import Flask, request, jsonify
import os
from RAG2     import RAG       # import the module once at process start (heavy imports happen here)

app = Flask(__name__)

def _to_bool(v, default=True):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes", "y", "t")

@app.route("/run", methods=["POST"])
def run_rag():
    """
    Expects JSON body (all fields optional):
    {
      "cq": "Competency question string",
      "start_rag": true,
      "core": "OntoDESIDECoreOntology",
      "class_count": 15,
      "op_count": 3,
      "dp_count": 5
    }
    """
    payload = request.get_json(silent=True) or {}

    competency_question = payload.get("cq", "What are the components of a product?")
    start_rag = _to_bool(payload.get("start_rag", True), default=True)
    core = payload.get("core", " ")
    try:
        class_count = int(payload.get("class_count", 15))
        op_count = int(payload.get("op_count", 3))
        dp_count = int(payload.get("dp_count", 5))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "class_count/op_count/dp_count must be integers"}), 400

    try:
        # Call into your RAG function exactly like your script did.
        # Adjust if RAG2.RAG returns prints instead of a result object (we still return whatever it returns).
        result = RAG.RAG(
            Query=competency_question,
            init_rag_flag=start_rag,
            class_count=class_count,
            op_count=op_count,
            dp_count=dp_count,
            core=core
        )
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        import traceback
        return (
            jsonify(
                {
                    "ok": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            ),
            500,
        )

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # For dev only. In production use gunicorn (instructions below).
    app.run(host="0.0.0.0", port=port, debug=False)
