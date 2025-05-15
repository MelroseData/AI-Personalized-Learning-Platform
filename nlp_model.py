import torch
from transformers import BertTokenizer, BertForSequenceClassification

class NLPModel:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)

    def preprocess(self, text):
        return self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    def predict(self, student_answer, professor_answer):
        inputs = self.preprocess([student_answer, professor_answer])
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        return torch.argmax(logits[0], dim=-1).item()