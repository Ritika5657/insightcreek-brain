# InsightCreek Brain 🧠  
*A simple ML-powered chatbot + lead prediction system*

## 📌 Overview
This project is a prototype chatbot brain that can:
- Predict lead conversion likelihood from sales notes.
- Analyze sentiment and extract keywords.
- Expose predictions through a **Flask API** for integration with backend/frontend.

It was built as part of a team project.  

## 🗂 Project Structure
```
insightcreek-brain/
│
├── code/
│   ├── analysis.py          # Text sentiment & keyword analysis
│   ├── chatbot_brain.py     # Chatbot intent handler
│   ├── inspect_errors.py    # Inspect misclassified samples
│   ├── predict_today.py     # Load model & predict single note
│   ├── server.py            # Flask API server
│   ├── test_calls.py        # Local test harness for chatbot_brain
│   ├── text_cleaner.py      # Custom text preprocessing transformer
│   └── train_model.py       # Training script
│
├── data/
│   └── clean_sales_data.csv # Training dataset
│
├── models/
│   ├── lead_pipeline.joblib # Trained ML model
│   └── misclassified.csv    # Inspect errors output
│
├── logs/                    # (empty, for logging if needed)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## ⚙️ Setup & Usage Guide

### 1. Create Virtual Environment
python -m venv .venv  
.venv\Scripts\activate      # On Windows

### 2. Install Dependencies
pip install -r requirements.txt

## 📊 Train the Model
To retrain the lead prediction model:
python code/train_model.py --data data/clean_sales_data.csv --out models/lead_pipeline.joblib --n-iter 3

- --data: path to CSV (must have note_text,label columns).  
- --out: where to save the trained pipeline.  
- --n-iter: number of hyperparameter search iterations.  

After training, the model is saved in models/lead_pipeline.joblib.

## 🔍 Inspect Errors
Check misclassified samples:
python code/inspect_errors.py --data data/clean_sales_data.csv --model models/lead_pipeline.joblib

Results are saved in models/misclassified.csv.

## 🤖 Test Chatbot Locally
python code/test_calls.py

Outputs sample interactions:  
- Greeting  
- Ask keywords  
- Ask sentiment  
- Predict lead likelihood  

## 🌐 Run API Server
Start Flask server:
python code/server.py

- Health check: GET http://127.0.0.1:5000/  
- Predict lead: POST http://127.0.0.1:5000/predict  
  Example body:
  { "note": "Customer liked the demo and asked about pricing." }

- Chatbot API: POST http://127.0.0.1:5000/api/chat  
  Example body:
  {
    "session_id": "s1",
    "message": "predict likelihood",
    "note": "Customer liked the demo and asked about pricing."
  }

## 🛠️ Next Steps
- Backend team → Wrap this API into main system.  
- Frontend team → Build UI and call the Flask endpoints.  


