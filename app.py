import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ATS Resume Scorer",
    page_icon="🎯",
    layout="centered"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background-color: #0f0f13; }
.block-container { padding-top: 2rem; max-width: 780px; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #2a2a4a;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
}
.hero h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
}
.hero p {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
}
.hero span { color: #60a5fa; }

/* Score card */
.score-card {
    background: linear-gradient(135deg, #1e3a5f, #1a2a4a);
    border: 1px solid #2d4a7a;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.score-number {
    font-size: 4rem;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.score-label {
    color: #94a3b8;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.score-verdict {
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 0.75rem;
}

/* Keyword pills */
.pill-container { display: flex; flex-wrap: wrap; gap: 8px; margin: 0.75rem 0; }
.pill {
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    font-family: 'DM Mono', monospace;
}
.pill-green { background: #14532d; color: #86efac; border: 1px solid #166534; }
.pill-red   { background: #450a0a; color: #fca5a5; border: 1px solid #7f1d1d; }

/* Section cards */
.section-card {
    background: #111118;
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.section-title {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.75rem;
}
.green { color: #4ade80; }
.red   { color: #f87171; }
.blue  { color: #60a5fa; }

/* Suggestion list */
.suggestion {
    background: #1a1a2e;
    border-left: 3px solid #60a5fa;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.88rem;
    color: #cbd5e1;
    line-height: 1.5;
}

/* Upload area */
.stFileUploader > div { border-radius: 12px; }
label { color: #94a3b8 !important; font-size: 0.85rem !important; }

/* Divider */
.divider { border: none; border-top: 1px solid #2a2a3a; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def extract_text_from_pdf(uploaded_file):
    """Extract all text from a PDF file."""
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def clean_text(text):
    """Lowercase, remove punctuation, normalize whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_keywords(text, top_n=40):
    """Extract top TF-IDF keywords from a text."""
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),   # unigrams + bigrams
        max_features=200
    )
    try:
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        sorted_idx = scores.argsort()[::-1]
        keywords = [feature_names[i] for i in sorted_idx[:top_n] if scores[i] > 0]
        return set(keywords)
    except:
        return set()


def compute_score(resume_text, jd_text):
    """Compute cosine similarity between resume and JD."""
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    try:
        tfidf = vectorizer.fit_transform([clean_text(resume_text), clean_text(jd_text)])
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return round(score * 100, 1)
    except:
        return 0.0


def get_keyword_overlap(resume_text, jd_text):
    """Return matched and missing keywords."""
    jd_keywords = extract_keywords(clean_text(jd_text), top_n=35)
    resume_clean = clean_text(resume_text)

    matched = set()
    missing = set()

    for kw in jd_keywords:
        if kw in resume_clean: #matching terms in resume
            matched.add(kw)
        else:
            missing.add(kw)

    return sorted(matched), sorted(missing)


def score_to_color(score):
    if score >= 75:
        return "#4ade80"#green
    elif score >= 50:
        return "#facc15"#yellow
    else:
        return "#f87171"#red


def score_to_verdict(score):
    if score >= 75:
        return "✅ Strong Match — Good to apply!"
    elif score >= 55:
        return "⚠️ Moderate Match — Needs some tuning"
    elif score >= 35:
        return "🔶 Weak Match — Significant gaps"
    else:
        return "❌ Poor Match — Major revision needed"


def generate_suggestions(missing_keywords, score):
    """Generate actionable improvement suggestions."""
    suggestions = []
    if score < 75 and missing_keywords:
        top_missing = missing_keywords[:8]
        kw_str = ", ".join([f'"{k}"' for k in top_missing[:5]])
        suggestions.append(
            f"🔑 <b>Add missing keywords</b> — Incorporate these JD terms naturally into your resume: {kw_str}"
        )
    if score < 60:
        suggestions.append(
            "📝 <b>Rewrite your summary/objective</b> — Mirror the language and priorities from the job description in your profile summary."
        )
    if any(kw in " ".join(missing_keywords) for kw in ["experience", "year", "senior", "lead", "manage"]):
        suggestions.append(
            "💼 <b>Highlight relevant experience</b> — The JD emphasizes seniority or leadership. Quantify your contributions with numbers and outcomes."
        )
    if any(kw in " ".join(missing_keywords) for kw in ["python", "sql", "java", "c++", "machine learning", "data", "cloud", "aws", "azure"]):
        suggestions.append(
            "🛠️ <b>Expand your skills section</b> — Several technical keywords from the JD are missing. Add them to your skills/tech stack section if you have them."
        )
    if score < 50:
        suggestions.append(
            "🎯 <b>Tailor this resume specifically for the role</b> — A generic resume won't pass ATS. Customize your bullet points to reflect this JD's exact requirements."
        )

    suggestions.append(
        "📄 <b>Use standard section headings</b> — Stick to headings like 'Experience', 'Education', 'Skills' so ATS parsers can read your resume correctly."
    )

    if score >= 75:
        suggestions.append(
            "✨ <b>You're in good shape!</b> — Focus on the few missing keywords and ensure your formatting is ATS-friendly (no tables or columns)."
        )
    return suggestions


#UI
st.markdown("""
<div class="hero">
    <h1>🎯 ATS Resume Scorer</h1>
    <p>Upload your resume and job description — get your <span>match score</span>, missing keywords, and tips to improve.</p>
</div>
""", unsafe_allow_html=True)
###
col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
with col2:
    jd_input_mode = st.radio("📋 Job Description Input", ["Paste Text", "Upload PDF"], horizontal=True)

if jd_input_mode == "Upload PDF":
    jd_file = st.file_uploader("Upload JD as PDF", type=["pdf"])
    jd_pasted = None
else:
    jd_pasted = st.text_area(
        "Paste the Job Description here",
        placeholder="Copy and paste the full job description text...",
        height=200
    )
    jd_file = None

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# Determine if we have both inputs ready
jd_ready = (jd_file is not None) or (jd_pasted and len(jd_pasted.strip()) > 50)
ready = resume_file and jd_ready

if ready:
    with st.spinner("Analyzing your resume..."):
        resume_text = extract_text_from_pdf(resume_file)

        if jd_file:
            jd_text = extract_text_from_pdf(jd_file)
        else:
            jd_text = jd_pasted.strip()

        if not resume_text:
            st.error("Couldn't extract text from your resume PDF. Make sure it's not a scanned image.")
            st.stop()
        if not jd_text:
            st.error("Couldn't read the job description. Try pasting the text instead.")
            st.stop()

        score = compute_score(resume_text, jd_text)
        matched_kws, missing_kws = get_keyword_overlap(resume_text, jd_text)
        suggestions = generate_suggestions(missing_kws, score)
        color = score_to_color(score)
        verdict = score_to_verdict(score)

    #scorecard
    st.markdown(f"""
    <div class="score-card">
        <div class="score-number" style="color:{color}">{score}%</div>
        <div class="score-label">ATS Match Score</div>
        <div class="score-verdict" style="color:{color}">{verdict}</div>
    </div>
    """, unsafe_allow_html=True)

    #matched keywords
    st.markdown(f"""
    <div class="section-card">
        <div class="section-title green">✅ Matched Keywords ({len(matched_kws)})</div>
        <div class="pill-container">
            {''.join([f'<span class="pill pill-green">{kw}</span>' for kw in matched_kws[:25]])}
        </div>
    </div>
    """, unsafe_allow_html=True)

    #missing keywords
    st.markdown(f"""
    <div class="section-card">
        <div class="section-title red">❌ Missing Keywords ({len(missing_kws)})</div>
        <div class="pill-container">
            {''.join([f'<span class="pill pill-red">{kw}</span>' for kw in missing_kws[:25]])}
        </div>
    </div>
    """, unsafe_allow_html=True)

    #suggestions
    st.markdown("""
    <div class="section-card">
        <div class="section-title blue">💡 Suggestions to Improve Your Resume</div>
    """, unsafe_allow_html=True)
    for s in suggestions:
        st.markdown(f'<div class="suggestion">{s}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    #stats footer
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("JD Keywords", len(matched_kws) + len(missing_kws))
    c2.metric("Matched", len(matched_kws))
    c3.metric("Missing", len(missing_kws))

elif resume_file and not jd_ready:
    st.info("Now paste the job description text (or upload the JD PDF) to see your score.")
elif jd_ready and not resume_file:
    st.info("Now upload your Resume PDF to see your score.")
else:
    st.markdown("""
    <div style="text-align:center; color:#4a4a6a; padding: 2rem 0; font-size:0.9rem;">
        Upload your resume and add the job description above to get started 👆
    </div>
    """, unsafe_allow_html=True)
