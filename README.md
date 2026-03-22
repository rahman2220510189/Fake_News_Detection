
# 🔍 Bangla Fake News Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green)
![React](https://img.shields.io/badge/React-18-61dafb)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Deployed-yellow)
![License](https://img.shields.io/badge/License-MIT-red)

> A three-class Bangla fake news detection system powered by Machine Learning, Transformers, and Large Language Models. The first system to detect **AI-generated fake news** in the Bangla language.

---

## 🚨 Problem Solved

With the rapid rise of social media and AI-generated content, fake news in Bangla has become a critical issue. Existing systems only classify news as real or fake — **none address AI-generated Bangla fake news**. This project:

- Detects **human-written fake news**
- Detects **AI-generated fake news** (first in Bangla NLP)
- Provides a **real-time API** for integration

---

## ✨ Features

- 🏷️ **Three-class classification**: `authentic` / `fake` / `ai_fake`
- 🤖 **Ensemble prediction**: SVM + BanglaBERT + Llama 70B majority voting
- ⚡ **Real-time API**: FastAPI backend deployed on HuggingFace Spaces
- 🌐 **Web Interface**: React.js frontend
- 🔄 **Groq API rotation**: 10-key rotation for rate limit handling
- 📊 **High accuracy**: 92% Macro F1 (SVM), 91.72% (BanglaBERT)

---

## 📊 Model Performance

| Model | Macro F1 | Auth F1 | Fake F1 | AI-Fake F1 |
|-------|----------|---------|---------|------------|
| SVM (TF-IDF) | **92.00%** 🏆 | 98% | 89% | 88% |
| BanglaBERT | 91.72% | 89% | 87% | 99% |
| Metadata Fusion | 91.69% | 89% | 87% | 99% |
| XLM-RoBERTa | 89.50% | 86% | 84% | 98% |
| Logistic Regression | 88.35% | 97% | 85% | 83% |
| BiLSTM | 87.29% | 84% | 81% | 97% |
| BLOOM LoRA | 86.50% | 80% | 76% | 97% |
| mBERT | 86.42% | 84% | 81% | 94% |
| Llama 3.2 1B LoRA | 77.81% | 77% | 70% | 86% |
| Naive Bayes | 71.04% | 67% | 78% | 68% |
| Zero-shot Llama 70B | 48.25% | 39% | 63% | 42% |

---

## 🛠️ Tech Stack

### Backend (FastAPI)
- **Framework**: FastAPI
- **ML Models**: SVM (scikit-learn), BanglaBERT (HuggingFace Transformers)
- **LLM**: Llama 3.3-70B via Groq API
- **Deployment**: HuggingFace Spaces (Docker)

### Frontend (React)
- **Framework**: React.js
- **HTTP Client**: Axios
- **Deployment**: Vercel

### Research & Training
- **Transformers**: BanglaBERT, XLM-RoBERTa, mBERT, BLOOM, Llama
- **Fine-tuning**: LoRA (PEFT)
- **NLP**: bnlp-toolkit (POS, NER)
- **GPU**: NVIDIA RTX 3050 (local), Google Colab T4

---

## 📁 Folder Structure

```
fake_news/
├── app/
│   └── main.py               # FastAPI backend
├── hf_space/                 # HuggingFace deployment
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── step1_preprocessing.ipynb
├── step2_metadata.ipynb
├── step3_ai_generation.ipynb
├── step4_balancing.ipynb
├── step5a_svm.ipynb
├── step5b_banglabert.ipynb
├── step5c_xlm_roberta.ipynb
├── step5d_mbert.ipynb
├── step5e_bloom_lora.ipynb
├── step5f_llama_lora.ipynb
├── step5g_bilstm.ipynb
├── step5h_zeroshot.ipynb
├── step6_confusion_matrix.ipynb
├── step7_nlp_features.ipynb
├── step8_metadata_fusion.ipynb
├── step9_word_cloud.ipynb
├── step10_ablation_study.ipynb
├── step11_error_analysis.ipynb
├── step12_ensemble.ipynb
├── step13_focal_loss.ipynb
├── step14_statistical_tests.ipynb
├── train_final.csv
├── val_final.csv
├── test_final.csv
├── svm_model.pkl
├── banglabert_best.pt
├── .env
└── .gitignore
```

---

## 🚀 Installation

### Backend (FastAPI)

```bash
# Clone repository
git clone https://github.com/rahman2220510189/Fake_News_Detection.git
cd Fake_News_Detection

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Add your GROQ_KEY_1 to GROQ_KEY_10 in .env

# Run locally
uvicorn app.main:app --reload
```

### Frontend (React)

```bash
# Clone frontend repository
git clone https://github.com/rahman2220510189/fake-news-frontend.git
cd fake-news-frontend

# Install dependencies
npm install

# Run locally
npm start
```

---

## 🔐 Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_KEY_1=your_groq_api_key_1
GROQ_KEY_2=your_groq_api_key_2
GROQ_KEY_3=your_groq_api_key_3
GROQ_KEY_4=your_groq_api_key_4
GROQ_KEY_5=your_groq_api_key_5
GROQ_KEY_6=your_groq_api_key_6
GROQ_KEY_7=your_groq_api_key_7
GROQ_KEY_8=your_groq_api_key_8
GROQ_KEY_9=your_groq_api_key_9
GROQ_KEY_10=your_groq_api_key_10
```

> Get free Groq API keys at [console.groq.com](https://console.groq.com)

---

## 📖 Usage Guide

### API Endpoints

**Base URL**: `https://rubayed001-fake-news.hf.space`

#### GET /
```json
{
  "message": "Bangla Fake News Detection API",
  "status": "running",
  "models": ["SVM", "BanglaBERT", "Llama-70B"]
}
```

#### POST /predict
```json
// Request
{
  "text": "বাংলা সংবাদের টেক্সট এখানে দিন..."
}

// Response
{
  "final_prediction": "authentic",
  "confidence": 0.92,
  "all_predictions": {
    "SVM": "authentic",
    "BanglaBERT": "authentic",
    "Llama_70B": "authentic"
  }
}
```

### Labels
| Label | Meaning |
|-------|---------|
| `authentic` | Real news ✅ |
| `fake` | Human-written fake news ❌ |
| `ai_fake` | AI-generated fake news 🤖 |

---

## 🌐 Live Demo

- **API**: [rubayed001-fake-news.hf.space](https://rubayed001-fake-news.hf.space/docs)
- **API Docs**: [rubayed001-fake-news.hf.space/docs](https://rubayed001-fake-news.hf.space/docs)

---

## 🔑 Key Findings

1. **AI-generated news is highly detectable** — BanglaBERT achieves 99% F1 for `ai_fake`
2. **AI articles are 5x shorter** — avg 62 tokens vs 314-323 for human-written
3. **Simple beats complex** — SVM (92%) outperforms all transformer & LLM models
4. **Monolingual > Multilingual** — BanglaBERT > XLM-RoBERTa > mBERT
5. **Zero-shot LLMs fail** — Llama 70B zero-shot: 48.25% (near random chance)

---

## 🤝 Contributing

Contributions are welcome!

```bash
# Fork the repository
# Create your feature branch
git checkout -b feature/AmazingFeature

# Commit your changes
git commit -m 'Add some AmazingFeature'

# Push to the branch
git push origin feature/AmazingFeature

# Open a Pull Request
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**MD Naymur Rahman**

- 🐙 GitHub: [@rahman2220510189](https://github.com/rahman2220510189)
- 🤗 HuggingFace: [@rubayed001](https://huggingface.co/rubayed001)

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{rahman2026bangla,
  title={Bangla Fake News Detection Using Machine Learning and Large Language Models},
  author={Rahman, MD Naymur},
  year={2026},
  publisher={GitHub},
  url={https://github.com/rahman2220510189/Fake_News_Detection}
}
```

---

<p align="center">
  Made with ❤️ for Bangla NLP Research
</p>
