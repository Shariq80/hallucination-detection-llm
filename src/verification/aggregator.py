import numpy as np

class ScoreAggregator:
    def __init__(self, config):
        # Weights (still useful for ranking, not final decision)
        self.sim_weight = config.get("aggregation", "similarity_weight")
        self.entail_weight = config.get("aggregation", "entailment_weight")

        # Thresholds (used in final decision)
        self.entail_threshold = config.get("aggregation", "entailment_threshold")
        self.contra_threshold = config.get("aggregation", "contradiction_threshold")
        self.sim_threshold = config.get("aggregation", "similarity_threshold")

    def aggregate(self, evidence_results, prefilter_similarity=True):

        # -----------------------------
        # 1. Assign similarity score
        # -----------------------------
        for evidence in evidence_results:
            evidence["similarity_score"] = evidence.get(
                "rerank_score",
                evidence.get("hybrid_score",
                             evidence.get("retriever_score",
                                          evidence.get("score", 0.0)))
            )

        # -----------------------------
        # 2. Filter weak evidence
        # -----------------------------
        valid_evidence = []
        for evidence in evidence_results:
            sim = evidence.get("similarity_score", 0.0)
            if not prefilter_similarity or sim >= self.sim_threshold:
                valid_evidence.append(evidence)

        if not valid_evidence:
            return {
                "final_score": 0.0,
                "avg_entailment": 0.0,
                "max_contradiction": 0.0,
                "label": "NOT_ENOUGH_INFO",
                "hallucinated": False,
                "best_evidence": None
            }

        # -----------------------------
        # 3. Compute scores (for logging only)
        # -----------------------------
        entailments = []
        contradictions = []
        scores = []

        best_score = float("-inf")
        best_evidence = None

        for evidence in valid_evidence:
            sim = evidence.get("similarity_score", 0.0)
            nli = evidence.get("nli_scores", {})

            entail = nli.get("entailment", 0.0)
            contra = nli.get("contradiction", 0.0)

            entailments.append(entail)
            contradictions.append(contra)

            # Score only for ranking best evidence
            score = self.sim_weight * sim + self.entail_weight * entail - contra
            scores.append(score)

            if score > best_score:
                best_score = score
                best_evidence = evidence

        # -----------------------------
        # 4. Strong evidence filtering
        # -----------------------------
        strong_evidence = [
            e for e in valid_evidence
            if e["similarity_score"] > 0.5 and e["nli_scores"].get("neutral", 0.0) < 0.95
        ]

        # If strong evidence exists → use it
        if strong_evidence:
            evidence_pool = strong_evidence
        else:
            evidence_pool = valid_evidence

        # -----------------------------
        # 5. Extract best signals
        # -----------------------------
        best_entail = max(e["nli_scores"].get("entailment", 0.0) for e in evidence_pool)
        best_contra = max(e["nli_scores"].get("contradiction", 0.0) for e in evidence_pool)

        # -----------------------------
        # 6. Final Decision (KEY FIX)
        # -----------------------------
        if best_entail >= self.entail_threshold:
            label = "SUPPORTED"
        elif best_contra >= self.contra_threshold:
            label = "REFUTED"
        else:
            label = "NOT_ENOUGH_INFO"

        # -----------------------------
        # 7. Return structured output
        # -----------------------------
        return {
            "final_score": float(np.mean(scores)),  # for debugging only
            "avg_entailment": float(np.mean(entailments)),
            "max_contradiction": float(np.max(contradictions)),
            "label": label,
            "hallucinated": label == "REFUTED",
            "best_evidence": best_evidence
        }