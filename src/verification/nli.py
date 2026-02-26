import torch 
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict

class NLIVerifier:
    def __init__(self, model_name="facebook/bart-large-mnli"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # MNLI label mapping
        self.label_mapping = {
            0: "contradiction",
            1: "neutral",
            2: "entailment"
        }

    def verify(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Perform NLI verification between premise and hypothesis.
        """
        # Tokenize premise and hypothesis as sentence pair
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        # Move tensors to device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # forward pass (no gradient computation for inference)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Convert logits to probabilities
        probabilities = F.softmax(logits, dim=-1)[0]

        contradiction_prob = probabilities[0].item()
        neutral_prob = probabilities[1].item()
        entailment_prob = probabilities[2].item()

        # Determine predicted label
        predicted_index = torch.argmax(probabilities).item()
        predicted_label = self.label_mapping[predicted_index]

        return {
            "entailment_probability": round(entailment_prob, 4),
            "contradiction_probability": round(contradiction_prob, 4),
            "neutral_probability": round(neutral_prob, 4),
            "predicted_label": predicted_label
        }

    def predict(self, claim, evidence):
        inputs = self.tokenizer(
            claim,
            evidence,
            return_tensors = "pt",
            truncation = True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=1)
        labels = ["contradiction", "neutral", "entailment"]

        return {
            label: probs[0][i].item()
            for i, label in enumerate(labels)
        }