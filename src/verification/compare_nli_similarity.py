import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
from similarity.similarity import SimilarityScorer
from verification.nli import NLIVerifier

def load_sample_data():
    """Returns sample claim/evidence pairs for comparison testing."""
    claims = [
        "The Eiffel Tower is in Paris.",                  # True, direct match
        "Water boils at 100 degrees Celsius.",              # True, scientific fact
        "Python is a snake and a language.",                # Partial true (hallucination component)
        "The moon is made of green cheese.",                # False/Hallucination
        "Albert Einstein invented the internet.",           # False/Hallucination
        "The Eiffel Tower is located in London.",           # False, contradiction
        "Python is primarily a high-level language."        # True, entailment
    ]
    
    evidences = [
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        "At sea level, water boils at 100 °C (212 °F).",
        "Python is a high-level, general-purpose programming language. Pythonidae is a family of nonvenomous snakes.",
        "The Moon is Earth's only natural satellite, composed of rock.",
        "The Internet was developed by DARPA and researchers like Vint Cerf and Bob Kahn.",
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        "Python is a high-level, general-purpose programming language."
    ]
    
    # Expected: Supported (Entailment), Contradiction, Neutral, etc.
    return claims, evidences

def compare_models():
    print("Loading Configuration...")
    config = Config()
    
    print("Initializing Similarity Scorer...")
    sim_scorer = SimilarityScorer(config)
    
    print("Initializing NLI Verifier...")
    nli_verifier = NLIVerifier(config)
    
    claims, evidences = load_sample_data()
    
    print("\n--- Comparing NLI vs Similarity Outputs ---\n")
    
    agreements = 0
    
    for i, (claim, evidence) in enumerate(zip(claims, evidences)):
        print(f"Sample {i+1}:")
        print(f"Claim:    {claim}")
        print(f"Evidence: {evidence}")
        
        # 1. Similarity Output (with threshold fixed to 0.70 from Week 4 tuning)
        sim_result = sim_scorer.predict(claim, evidence, threshold=0.70)
        sim_score = sim_result["score"]
        sim_pred = sim_result["prediction"]
        
        # 2. NLI Output
        nli_result = nli_verifier.verify(evidence, claim)
        nli_label = nli_result["predicted_label"]
        nli_detail = f"Entail: {nli_result['entailment_probability']:.2f}, Contradict: {nli_result['contradiction_probability']:.2f}, Neutral: {nli_result['neutral_probability']:.2f}"
        
        # Map NLI to Supported/Hallucinated for direct comparison
        nli_binary = "Supported" if nli_label == "entailment" else "Hallucinated"
        
        match = "Yes" if sim_pred == nli_binary else "No"
        if match == "Yes": agreements += 1
        
        print("-" * 50)
        print(f"Similarity Score: {sim_score:.4f} -> {sim_pred}")
        print(f"NLI Prediction:   {nli_label} -> {nli_binary}")
        print(f"NLI Details:      {nli_detail}")
        print(f"Models Agree?     {match}\n")

    print("=" * 50)
    print(f"Summary: The models agreed on {agreements} out of {len(claims)} samples ({(agreements/len(claims))*100:.1f}% agreement).")
    print("=" * 50)

if __name__ == "__main__":
    compare_models()
