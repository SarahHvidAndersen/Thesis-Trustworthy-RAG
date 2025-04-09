from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
model = AutoModelForSequenceClassification.from_pretrained("tasksource/deberta-small-long-nli")
tokenizer = AutoTokenizer.from_pretrained("tasksource/deberta-small-long-nli")

pair = ("A cat is a small domesticated carnivorous mammal.", 
        "A cat is a small domesticated carnivorous mammal.")




inputs = tokenizer(pair[0], pair[1], return_tensors="pt", truncation=True)
with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    print("Entailment probability:", probs[0, 2].item())  # index 2 is usually entail
