from similarity import SimilarityScorer

def test_pipeline():
    print("Testing SimilarityScorer Pipeline...")
    scorer = SimilarityScorer()

    claims = [
        "The Eiffel Tower is in Berlin.",
        "Water boils at 100 degrees Celsius.",
        "Python is a programming language."
    ]
    evidences = [
        "The Eiffel Tower is located in Paris, France.",
        "At sea level, water boils at 100 °C (212 °F).",
        "Python is a high-level, general-purpose programming language."
    ]

    print("\n--- Testing Single Item ---")
    score = scorer.score(claims[0], evidences[0])
    print(f"Claim: {claims[0]}")
    print(f"Evidence: {evidences[0]}")
    print(f"Score: {score:.4f}")

    print("\n--- Testing Batch Pipeline ---")
    batch_scores = scorer.score_batch(claims, evidences)
    for c, e, s in zip(claims, evidences, batch_scores):
        print(f"Claim: {c} | Score: {s:.4f}")

    print("\n--- Testing Prediction Pipeline (Threshold 0.5) ---")
    predictions = scorer.predict_batch(claims, evidences, threshold=0.5)
    for c, p in zip(claims, predictions):
        print(f"Claim: {c} | Pred: {p['prediction']} ({p['score']:.4f})")

if __name__ == "__main__":
    test_pipeline()
