from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
from huggingface_hub import hf_hub_download
import os
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from groq import Groq
import itertools

# Load .env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Groq keys rotation
GROQ_KEYS = [
    os.getenv(f"GROQ_KEY_{i}") for i in range(1, 11)
]
GROQ_KEYS = [k for k in GROQ_KEYS if k]
key_cycle = itertools.cycle(GROQ_KEYS)

def get_groq_client():
    return Groq(api_key=next(key_cycle))

app = FastAPI(title="Bangla Fake News Detection API")

# Labels
id2label = {0: 'authentic', 1: 'fake', 2: 'ai_fake'}

# Load SVM from HuggingFace
svm_path = hf_hub_download(
    repo_id="rubayed001/bangla-fake-news-banglabert",
    filename="svm_model.pkl"
)
svm_model = joblib.load(svm_path)

# Load BanglaBERT from HuggingFace
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
bert_model = AutoModelForSequenceClassification.from_pretrained(
    "csebuetnlp/banglabert", num_labels=3)
bert_path = hf_hub_download(
    repo_id="rubayed001/bangla-fake-news-banglabert",
    filename="banglabert_best.pt"
)
bert_model.load_state_dict(
    torch.load(bert_path, map_location=device))
bert_model = bert_model.to(device)
bert_model.eval()

print(f"✅ Models loaded! Device: {device}")
print(f"✅ Groq keys loaded: {len(GROQ_KEYS)}")

class NewsInput(BaseModel):
    text: str

def bert_predict(text):
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = bert_model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device)
        )
    proba = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred = np.argmax(proba)
    return id2label[pred], float(proba[pred])

def groq_predict(text):
    prompt = f"""তুমি একজন বাংলা সংবাদ যাচাইকারী।
নিচের সংবাদটি পড়ো এবং শুধুমাত্র একটি শব্দে উত্তর দাও:
- authentic (যদি আসল সংবাদ হয়)
- fake (যদি মানুষের লেখা ভুয়া হয়)
- ai_fake (যদি AI দিয়ে তৈরি ভুয়া হয়)

সংবাদ: {text[:500]}

উত্তর:"""

    for _ in range(len(GROQ_KEYS)):
        try:
            client = get_groq_client()
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10
            )
            result = response.choices[0].message.content.strip().lower()
            if 'ai_fake' in result:
                return 'ai_fake'
            elif 'fake' in result:
                return 'fake'
            else:
                return 'authentic'
        except Exception:
            continue
    return 'authentic'  # fallback

@app.get("/")
def home():
    return {
        "message": "Bangla Fake News Detection API",
        "status": "running",
        "models": ["SVM", "BanglaBERT", "Llama-70B"]
    }

@app.post("/predict")
def predict(news: NewsInput):
    text = news.text

    # SVM
    svm_pred = svm_model.predict([text])[0]

    # BanglaBERT
    bert_pred, bert_conf = bert_predict(text)

    # Groq
    try:
        groq_pred = groq_predict(text)
    except:
        groq_pred = bert_pred

    # Majority voting
    predictions = [svm_pred, bert_pred, groq_pred]
    final_pred = max(set(predictions), key=predictions.count)

    return {
        "final_prediction": final_pred,
        "confidence": bert_conf,
        "all_predictions": {
            "SVM": svm_pred,
            "BanglaBERT": bert_pred,
            "Llama_70B": groq_pred
        }
    }