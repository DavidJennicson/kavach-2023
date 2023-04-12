

from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load saved BERT model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = 'bertmod'
model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='models',do_lower_case=True)

# Example text to classify
text = "Rahul Gandhi is the best says Mamta"

# Tokenize the text and convert to input IDs and attention masks
inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)

# Make prediction using BERT model
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs[0]
    predicted_class = torch.argmax(logits, dim=1).item()

print(f"Predicted class: {predicted_class}")
find_party(predicted_class)

