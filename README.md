# Federated Sentiment Analysis with Flower + DistilBERT

This project implements a **Federated Learning** setup using the **Flower** framework and **DistilBERT** from Hugging Face for **sentiment analysis** on the `twitter_training.csv` dataset. It simulates **3 clients** collaboratively training a model without sharing raw data.

---

##  Project Structure
```
federated_sentiment_flower/
│
├── twitter_training.csv
├── split_clients.py          # Split dataset into 3 clients
├── train_utils.py            # Model, tokenizer, train, evaluate functions
├── server.py                 # Flower server (FedAvg)
├── client.py                 # Flower NumPyClient
├── app.py                    # Flask API for predictions
├── visualize.py              # Accuracy/Loss plots
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 2️. Install Requirements
```bash
pip install torch torchvision transformers datasets pandas scikit-learn matplotlib flask flwr
```

### 3️. Place Dataset
Download and save `twitter_training.csv` (from Kaggle) in the project root directory.

Expected columns:
```
text,label
```

---

## Step 1: Split Dataset for Clients
```bash
python split_clients.py
```
Creates 3 files:
```
client_data_1.csv
client_data_2.csv
client_data_3.csv
```

---

##  Step 2: Run Federated Learning

### Terminal 1 — Start Server
```bash
python server.py
```

### Terminals 2, 3, 4 — Start Clients
```bash
python client.py 1
python client.py 2
python client.py 3
```

After 3 training rounds, the model will be aggregated.

---

##  Step 3: Save Model
Save the final aggregated model as:
```
final_model/
```
You can save the trained model using the Hugging Face `save_pretrained()` method in your training script.

---

##  Step 4: Run Flask API
```bash
python app.py
```

### Test API
```bash
curl -X POST -H "Content-Type: application/json" -d '{"text":"I love this project!"}' http://127.0.0.1:5000/predict
```
Expected output:
```json
{"sentiment": "Positive"}
```

---

##  Step 5: Visualize Metrics
```bash
python visualize.py
```
Generates:
- `accuracy_plot.png`
- `loss_plot.png`

---

##  Debug Tips
- Ensure all 3 clients are connected before training begins.
- Verify CSV has correct columns: `text`, `label`.
- If you get CUDA errors, remove `.to('cuda')` for CPU-only systems.
- Run Flask API only **after** training and saving the model.

---

##  Future Improvements
- Add an orchestrator to automatically run server + clients.
- Store metrics logs for dynamic visualization.
- Extend for more clients or a multilingual dataset.

---

**Author:** Aditya Pramod Parab  
**Project:** Federated Sentiment Analysis using Flower + DistilBERT  
**Clients:** 3  
**Dataset:** `twitter_training.csv`

