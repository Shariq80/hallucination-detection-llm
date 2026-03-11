import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from similarity import SimilarityScorer

def load_sample_data():
    """Returns sample FEVER-like claim/evidence pairs with labels for tuning."""
    # 1 = Supported (Entailment), 0 = Hallucinated (Contradiction/Neutral)
    claims = [
        "The Eiffel Tower is in Paris.",           # Supported
        "Water boils at 100 degrees Celsius.",       # Supported
        "Python is a snake and a language.",         # Partially supported/Hallucinated
        "The moon is made of green cheese.",         # Hallucinated
        "Albert Einstein invented the internet."     # Hallucinated
    ]
    
    evidences = [
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        "At sea level, water boils at 100 °C (212 °F).",
        "Python is a high-level, general-purpose programming language.",
        "The Moon is Earth's only natural satellite.",
        "The Internet was developed by DARPA and researchers like Vint Cerf and Bob Kahn."
    ]
    
    labels = [1, 1, 0, 0, 0]
    return claims, evidences, labels

def tune_threshold(scorer, claims, evidences, labels, start=0.1, end=0.9, step=0.05):
    """Iterates through thresholds to find the best F1 score."""
    print("Computing similarity scores for tuning dataset...")
    scores = scorer.score_batch(claims, evidences)
    
    best_f1 = -1.0
    best_threshold = 0.5
    results = []

    print("\nStarting threshold search...")
    for t in np.arange(start, end + step, step):
        t = round(t, 2)
        # Predict: 1 if score >= t (Supported), else 0 (Hallucinated)
        predictions = [1 if s >= t else 0 for s in scores]
        
        # Calculate metrics using zero_division=0 to handle cases where no positive predictions are made
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        acc = accuracy_score(labels, predictions)
        
        results.append((t, f1, precision, recall, acc))
        print(f"Threshold: {t:.2f} | F1: {f1:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | Acc: {acc:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1, results

if __name__ == "__main__":
    scorer = SimilarityScorer()
    claims, evidences, labels = load_sample_data()
    
    best_t, best_f1, _ = tune_threshold(scorer, claims, evidences, labels)
    
    print("\n" + "="*40)
    print(f"Optimal Similarity Threshold: {best_t:.2f}")
    print(f"Best F1 Score: {best_f1:.4f}")
    print("="*40)
    print("\nUday's Week 4 Task: Define similarity threshold - COMPLETE")
