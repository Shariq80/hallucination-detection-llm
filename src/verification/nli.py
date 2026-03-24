from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import re


class NLIVerifier:

    def __init__(self, config):

        model_name = config.get("models", "nli_model")

        print(f"Loading NLI model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.device = self._get_device(config)
        self.model.to(self.device)
        self.model.eval()

    def _get_device(self, config):
        device = config.get("nli", "device")

        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return torch.device(device)

    def _split_sentences(self, text):
        # ✅ Simple and effective sentence splitter
        sentences = re.split(r'(?<=[.!?]) +', text)
        return [s.strip() for s in sentences if s.strip()]

    def predict(self, claim, evidence):

        sentences = self._split_sentences(evidence)

        best_scores = {
            "contradiction": 0.0,
            "neutral": 0.0,
            "entailment": 0.0
        }

        for sentence in sentences:

            inputs = self.tokenizer(
                claim,
                sentence,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)[0]

            # DeBERTa mapping
            contradiction = probs[0].item()
            neutral = probs[1].item()
            entailment = probs[2].item()

            # ✅ Take best across sentences
            best_scores["contradiction"] = max(best_scores["contradiction"], contradiction)
            best_scores["neutral"] = max(best_scores["neutral"], neutral)
            best_scores["entailment"] = max(best_scores["entailment"], entailment)

        return best_scores