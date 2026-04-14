"""
Flask web server for the Hallucination Detection UI.
Provides API endpoints for multi-model claim verification
and pre-computed experiment results.
"""

import sys
import os
import json
from pathlib import Path
from flask import Flask, render_template, jsonify, request

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.pipeline import HallucinationPipeline
from src.claim_generator import generate_claims

app = Flask(__name__)

# NLI models to compare
NLI_MODELS = [
    "facebook/bart-large-mnli",
    "roberta-large-mnli",
    "typeform/distilbert-base-uncased-mnli"
]

CONFIG_PATH = "configs/config_small.yaml"


# ───────────────────────── Routes ─────────────────────────

@app.route("/")
def index():
    """Serve the main dashboard."""
    return render_template("index.html")


@app.route("/api/generate_claims", methods=["POST"])
def api_generate_claims():
    """
    Generate claims based on a topic using the Gemini LLM.

    Request body: { "topic": "some topic", "num_claims": 5 }
    """
    data = request.get_json()
    topic = data.get("topic", "").strip()
    num_claims = data.get("num_claims", 5)

    if not topic:
        return jsonify({"error": "No topic provided"}), 400

    try:
        num_claims = int(num_claims)
        num_claims = max(1, min(num_claims, 10))  # limit between 1 and 10
    except ValueError:
        return jsonify({"error": "Invalid num_claims"}), 400

    print(f"\n========================================")
    print(f"Generating {num_claims} claims for topic: '{topic}'")
    print(f"========================================")

    try:
        claims = generate_claims(topic, n_claims=num_claims)
        return jsonify({"claims": claims})
    except Exception as e:
        print(f"Error generating claims: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/verify", methods=["POST"])
def verify_claim():
    """
    Run a claim through all NLI models and return comparison results.

    Request body: { "claim": "some statement" }
    """
    data = request.get_json()
    claim = data.get("claim", "").strip()
    top_k = data.get("top_k", 5)

    # Clamp to safe range
    top_k = max(1, min(top_k, 20))

    if not claim:
        return jsonify({"error": "No claim provided"}), 400

    model_results = []

    for model_name in NLI_MODELS:
        try:
            print(f"\n{'='*50}")
            print(f"Running model: {model_name}")
            print(f"{'='*50}")

            pipeline = HallucinationPipeline(
                config_path=CONFIG_PATH,
                nli_model_override=model_name
            )
            # Override the evidence count
            pipeline.retriever.top_k = top_k

            result = pipeline.verify(claim)

            # Extract the atomic result (we use the first one)
            atomic = result["atomic_results"][0]
            final = atomic["final_result"]
            evidence = atomic["evidence"]

            model_results.append({
                "model_name": model_name,
                "label": final.get("label", "UNKNOWN"),
                "hallucinated": final.get("hallucinated", False),
                "final_score": final.get("final_score", 0),
                "nli_scores": {
                    "entailment": final.get("best_entailment", final.get("avg_entailment", 0)),
                    "contradiction": final.get("best_contradiction", final.get("max_contradiction", 0)),
                    "neutral": 1.0 - final.get("best_entailment", final.get("avg_entailment", 0))
                              - final.get("best_contradiction", final.get("max_contradiction", 0))
                },
                "similarity_score": max(
                    (e.get("similarity_score", 0) for e in evidence), default=0
                ),
                "evidence": [
                    {
                        "title": e["title"],
                        "text": e["text"],
                        "retriever_score": e.get("retriever_score", 0),
                        "similarity_score": e.get("similarity_score", 0),
                        "nli_scores": e.get("nli_scores", {})
                    }
                    for e in evidence
                ],
                "raw_final_result": final
            })

        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            model_results.append({
                "model_name": model_name,
                "label": "ERROR",
                "hallucinated": False,
                "final_score": 0,
                "nli_scores": {"entailment": 0, "contradiction": 0, "neutral": 0},
                "similarity_score": 0,
                "evidence": [],
                "error": str(e)
            })

    return jsonify({
        "claim": claim,
        "model_results": model_results
    })


@app.route("/api/results", methods=["GET"])
def get_experiment_results():
    """
    Return pre-computed experiment results from the results/ directory.
    """
    results_dir = Path("results")
    output = {}

    if not results_dir.exists():
        return jsonify(output)

    for json_file in sorted(results_dir.glob("*.json")):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            key = json_file.stem  # e.g., 'exp1_baseline'
            output[key] = {
                "config": data.get("config", ""),
                "metrics": data.get("metrics", {}),
                "results": [
                    {
                        "claim": r["claim"],
                        "true_label": r["true_label"],
                        "predicted_label": r["predicted_label"]
                    }
                    for r in data.get("results", [])
                ]
            }
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    return jsonify(output)


# ───────────────────────── Main ─────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Hallucination Detector — Web UI")
    print("  http://localhost:5000")
    print("=" * 60 + "\n")

    app.run(debug=False, host="0.0.0.0", port=5000)
