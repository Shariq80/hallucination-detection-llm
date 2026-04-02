# src/verify_claim.py
from src.pipeline.pipeline import HallucinationPipeline

class ClaimVerifier:
    def __init__(self, config_path="configs/default.yaml"):
        self.pipeline = HallucinationPipeline(config_path)

    def verify_claims(self, claims):
        results = []
        for claim in claims:
            result = self.pipeline.verify(claim)
            results.append(result)
        return results