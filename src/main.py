# main.py
import os
import argparse

from src.claim_generator import generate_claims
from src.pipeline.pipeline import HallucinationPipeline

# Set your Google AI API key
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_AI_STUDIO_API_KEY"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    print(f"\nUsing config: {args.config}")

    topic = input("Enter a topic to generate claims: ")
    num_claims = int(input("Number of claims to generate: "))

    # Step 1: Generate claims
    claims = generate_claims(topic, n_claims=num_claims)

    print("\nGenerated Claims:")
    for c in claims:
        print("-", c)

    # NLI models to test
    nli_models = [
        "facebook/bart-large-mnli",
        "roberta-large-mnli",
        "typeform/distilbert-base-uncased-mnli"
    ]

    print("\n==============================")
    print("MULTI-MODEL VERIFICATION")
    print("==============================")

    # Step 2: Loop over models
    for model_name in nli_models:
        print("\n========================================")
        print(f"RUNNING WITH MODEL: {model_name}")
        print("========================================")

        pipeline = HallucinationPipeline(
            config_path=args.config,
            nli_model_override=model_name
        )

        for claim in claims:
            result = pipeline.verify(claim)

            print("\n--- RESULT ---")
            print(f"Claim: {claim}")

            for atomic in result["atomic_results"]:
                label = atomic["final_result"]["label"]
                print(f"Prediction ({model_name}): {label}")

            print("---------------------------")


if __name__ == "__main__":
    main()