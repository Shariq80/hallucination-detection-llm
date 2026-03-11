class ScoreAggregator:
    def __init__(self, config):
        self.sim_weight = config.get("aggregation", "similarity_weight")
        self.entail_weight = config.get("aggregation", "entailment_weight")

    def aggregate(self, similarity_score, nli_scores):
        entail = nli_scores["entailment"]
        contra = nli_scores["contradiction"]

        final_score = (self.sim_weight * similarity_score + self.entail_weight * entail)

        hallucinated = contra > entail

        return {
            "final_score": final_score,
            "hallucinated": hallucinated
        }