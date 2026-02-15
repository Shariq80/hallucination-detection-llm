import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class NLIVerifier:
    def __init__(self, model_name="facebook/bart-large-mnli"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

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