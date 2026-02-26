class ScoreAggregator:
    def aggregate(self, similarity_score, nli_scores):
        entail = nli_scores["entailment"]
        contra = nli_scores["contradiction"]

        final_score = (0.4 * similarity_score) + (0.6 * entail)

        hallucinated = contra > entail

        return {
            "final_score": final_score,
            "hallucinated": hallucinated
        }