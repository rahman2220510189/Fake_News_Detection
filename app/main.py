# fake_news/app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from groq import Groq
import os

app = FastAPI(title="Bangla Fake News Detection API")

# Labels
label2id = {'authentic': 0, 'fake': 1, 'ai_fake': 2}
id2label = {0: 'authentic', 1: 'fake', 2: 'ai_fake'}

# Load SVM
svm_model = joblib.load('../svm_model.pkl')

# Load BanglaBERT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
bert_model = AutoModelForSequenceClassification.from_pretrained(
    "csebuetnlp/banglabert", num_labels=3)
bert_model.load_state_dict(torch.load('../banglabert_best.pt'))
bert_model = bert_model.to(device)
bert_model.eval()

# Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

    response = groq_client.chat.completions.create(
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

@app.get("/")
def home():
    return {"message": "Bangla Fake News Detection API", "status": "running"}

@app.post("/predict")
def predict(news: NewsInput):
    text = news.text

    # SVM prediction
    svm_pred = svm_model.predict([text])[0]

    # BanglaBERT prediction
    bert_pred, bert_conf = bert_predict(text)

    # Groq prediction
    try:
        groq_pred = groq_predict(text)
    except:
        groq_pred = bert_pred

    # Majority voting
    predictions = [svm_pred, bert_pred, groq_pred]
    final_pred = max(set(predictions), key=predictions.count)

    return {
        "text": text[:100] + "...",
        "final_prediction": final_pred,
        "confidence": bert_conf,
        "all_predictions": {
            "SVM": svm_pred,
            "BanglaBERT": bert_pred,
            "Llama_70B": groq_pred
        },
        "best_model": "BanglaBERT"
    }