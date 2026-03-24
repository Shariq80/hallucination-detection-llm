class ScoreAggregator:

    def __init__(self, config):

        self.sim_weight = config.get("aggregation", "similarity_weight")
        self.entail_weight = config.get("aggregation", "entailment_weight")

        self.entail_threshold = self._safe_get(config, "aggregation", "entailment_threshold", 0.5)
        self.contra_threshold = self._safe_get(config, "aggregation", "contradiction_threshold", 0.5)
        self.contra_penalty = self._safe_get(config, "aggregation", "contradiction_penalty", 0.5)

        # ✅ Optional similarity filtering
        self.sim_threshold = self._safe_get(config, "aggregation", "similarity_threshold", 0.0)

    def _safe_get(self, config, section, key, default):
        try:
            value = config.get(section, key)
            return default if value is None else value
        except:
            return default

    def aggregate(self, evidence_results):

        best_score = float("-inf")
        best_evidence = None

        for evidence in evidence_results:

            sim = evidence.get("similarity_score", 0.0)

            # ✅ Skip weak evidence
            if sim < self.sim_threshold:
                continue

            nli = evidence.get("nli_scores", {})

            entail = nli.get("entailment", 0.0)
            contra = nli.get("contradiction", 0.0)

            # ✅ Improved scoring
            score = (
                self.sim_weight * sim +
                self.entail_weight * entail -
                self.contra_penalty * contra
            )

            if score > best_score:
                best_score = score
                best_evidence = evidence

        # ✅ Handle no valid evidence
        if not best_evidence:
            return {
                "final_score": 0.0,
                "best_entailment": 0.0,
                "best_contradiction": 0.0,
                "label": "NOT_ENOUGH_INFO",
                "hallucinated": False,
                "best_evidence": None
            }

        # ✅ Use SAME evidence for decision
        best_nli = best_evidence.get("nli_scores", {})
        best_entail = best_nli.get("entailment", 0.0)
        best_contra = best_nli.get("contradiction", 0.0)

        # ✅ Decision logic
        if best_entail >= self.entail_threshold:
            label = "SUPPORTED"
        elif best_contra >= self.contra_threshold:
            label = "REFUTED"
        else:
            label = "NOT_ENOUGH_INFO"

        return {
            "final_score": best_score,
            "best_entailment": best_entail,
            "best_contradiction": best_contra,
            "label": label,
            "hallucinated": label == "REFUTED",
            "best_evidence": best_evidence
        }