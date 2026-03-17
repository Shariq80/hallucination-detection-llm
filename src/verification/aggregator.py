class ScoreAggregator:

    def __init__(self, config):

        self.sim_weight = config.get("aggregation", "similarity_weight")
        self.entail_weight = config.get("aggregation", "entailment_weight")

    def aggregate(self, evidence_results):

        best_entail = 0
        best_contra = 0
        best_score = 0

        for evidence in evidence_results:

            sim = evidence["similarity"]
            entail = evidence["nli"]["entailment"]
            contra = evidence["nli"]["contradiction"]

            score = (self.sim_weight * sim + self.entail_weight * entail)

            best_score = max(best_score, score)
            best_entail = max(best_entail, entail)
            best_contra = max(best_contra, contra)

        hallucinated = best_contra > best_entail

        return {
            "final_score": best_score,
            "best_entailment": best_entail,
            "best_contradiction": best_contra,
            "hallucinated": hallucinated
        }