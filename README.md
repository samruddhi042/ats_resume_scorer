# 🎯 ATS Resume Scorer

A Streamlit web app that scores how well your resume matches a job description using **TF-IDF vectorization** and **cosine similarity** — the same technique used by real ATS systems.

---

## Features

- 📄 Upload resume as PDF
- 📋 Paste job description as text **or** upload as PDF
- 📊 Get an ATS match score (0–100%)
- ✅ See which keywords from the JD are already in your resume
- ❌ See which keywords are missing
- 💡 Get actionable suggestions to improve your resume

---

## Tech Stack

- **Streamlit** — frontend UI
- **pdfplumber** — PDF text extraction
- **scikit-learn** — TF-IDF vectorization + cosine similarity

---

## Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python -m streamlit run app.py
```

---

## How It Works

1. Resume PDF is parsed and cleaned (lowercased, punctuation removed)
2. JD is accepted as pasted text or PDF
3. TF-IDF vectors are computed for both texts using bigrams
4. Cosine similarity between resume and JD vectors = match score
5. Top JD keywords are extracted and checked against resume text
6. Suggestions are generated based on score + missing keywords

---

## Why TF-IDF + Cosine Similarity instead of spaCy?

| | spaCy | TF-IDF + Cosine |
|---|---|---|
| What it does | Understands *meaning* of language | Matches *important words* statistically |
| How ATS systems actually work | ❌ Too complex for most ATS | ✅ This is literally what ATS uses |
| Needs model download | Yes | No, works out of the box |
| Speed | Slower | Very fast |
| Best for | Chatbots, summarization, Named Entity Recognition | Document similarity, keyword matching |

> Real ATS tools like Greenhouse, Workday, Taleo — they don't understand your resume like a human. They just count and match keywords statistically. TF-IDF + cosine is exactly that. So this scorer actually mimics real ATS behaviour more accurately than spaCy would have.

---

## Author

**Samruddhi Chavan** — B.Tech CSE (Data Science), VIT Pune
