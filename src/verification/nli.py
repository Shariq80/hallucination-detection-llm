# src/verification/nli.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import re

class NLIVerifier:
    def __init__(self, config, model_name=None):
        # Use the override model if provided, else read from config
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = config.get("models", "nli_model")

        print(f"Loading NLI model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        self.device = self._get_device(config)
        self.model.to(self.device)
        self.model.eval()

    def _normalize(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    def _get_device(self, config):
        device = config.get("nli", "device")
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _split_sentences(self, text):
        sentences = re.split(r'(?<=[.!?]) +', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def predict(self, claim, evidence):
        sentences = self._split_sentences(evidence)
        if not sentences:
            sentences = [evidence]

        best_scores = {"contradiction": 0.0, "neutral": 0.0, "entailment": 0.0}

        for sentence in sentences:
            inputs = self.tokenizer(
                claim,
                sentence,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)[0]

            contradiction, neutral, entailment = probs[0].item(), probs[1].item(), probs[2].item()

            best_scores["contradiction"] = max(best_scores["contradiction"], contradiction)
            best_scores["neutral"] = max(best_scores["neutral"], neutral)
            best_scores["entailment"] = max(best_scores["entailment"], entailment)

        return best_scores