from flask import Flask, request, jsonify
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch


app = Flask(__name__)
model = DistilBertForSequenceClassification.from_pretrained('final_model')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return jsonify({'sentiment': 'Positive' if pred == 1 else 'Negative'})


if __name__ == '__main__':
    app.run(debug=True)