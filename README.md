# 🎯 ATS Resume Scorer

A Streamlit web app that scores how well your resume matches a job description using **TF-IDF vectorization** and **cosine similarity** — the same technique used by real ATS systems.

## Features
- 📄 Upload resume + job description as PDFs
- 📊 Get an ATS match score (0–100%)
- ✅ See which keywords from the JD are already in your resume
- ❌ See which keywords are missing
- 💡 Get actionable suggestions to improve your resume

## Tech Stack
- **Streamlit** — frontend UI
- **pdfplumber** — PDF text extraction
- **scikit-learn** — TF-IDF vectorization + cosine similarity

## Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## How It Works
1. Both PDFs are parsed and cleaned (lowercased, punctuation removed)
2. TF-IDF vectors are computed for both texts using bigrams
3. Cosine similarity between resume and JD vectors = match score
4. Top JD keywords are extracted and checked against resume text
5. Suggestions are generated based on score + missing keywords

## Author
**Samruddhi Chavan** — B.Tech CSE (Data Science), VIT Pune
