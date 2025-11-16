import streamlit as st
import re
from collections import Counter
import pandas as pd

st.set_page_config(page_title="Next Word Prediction Model", layout="wide")

st.markdown("""
<style>

[data-testid="stSidebar"] {
    position: relative;
    overflow: hidden;
    padding-top: 20px;
    background: #002b55;
}

/* Particle container */
[data-testid="stSidebar"]::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 220%;
    height: 200%;
    background-image: radial-gradient(circle, rgba(255,255,255,0.25) 2px, transparent 2px);
    background-size: 50px 50px;
    animation: particleFloat 18s linear infinite;
    opacity: 0.35;
}

/* Second particle layer for depth */
[data-testid="stSidebar"]::after {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 160%;
    height: 180%;
    background-image: radial-gradient(circle, rgba(0,255,255,0.35) 2px, transparent 2px);
    background-size: 70px 70px;
    animation: particleFloat2 28s linear infinite;
    opacity: 0.28;
}

@keyframes particleFloat {
    0% { transform: translate(0,0); }
    50% { transform: translate(-15%, -10%); }
    100% { transform: translate(0,0); }
}

@keyframes particleFloat2 {
    0% { transform: translate(0,0); }
    50% { transform: translate(10%, 12%); }
    100% { transform: translate(0,0); }
}

/* Ensure sidebar elements stay visible above animation */
[data-testid="stSidebar"] * {
    position: relative;
    z-index: 10;
    color: white !important;
}

/* Student Box Styling */
.student-box {
    background: rgba(255,255,255,0.12);
    padding: 14px;
    border-radius: 12px;
    margin-bottom: 18px;
    text-align: center;
            
}

.student-title {
    font-size: 20px;
    font-weight: bold;
    color: #00eaff;
            
}

.separator {
    border-bottom: 1px solid rgba(255,255,255,0.3);
    margin: 16px 0;
}

</style>
""", unsafe_allow_html=True)

raw_sentences = [
    "The movie was absolutely fantastic and the acting was brilliant",
    "I really enjoyed the storyline and the performances",
    "The direction was weak but the soundtrack was amazing",
    "The film had great visuals but the plot was very slow",
    "I would definitely recommend this movie to my friends",
    "The ending was disappointing and very predictable",
    "The actors did a wonderful job in this emotional drama",
    "The movie was too long and a bit boring in the middle",
    "I loved the cinematography and the background score",
    "Overall it was a decent movie with some great moments",
]

# PREPROCESSING FUNCTIONS (stopwords, punctuation, tokenize)


stopwords = {
    "the", "is", "am", "are", "a", "an", "and", "or", "of", "to", "in",
    "was", "were", "it", "this", "that", "for", "on", "with", "as",
    "but", "be", "by", "at", "from", "very", "too"
}

def preprocess(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-z\s]", "", sentence)
    words = [w for w in sentence.split() if w not in stopwords]
    return words

tokenized = [preprocess(x) for x in raw_sentences]


# BUILD BIGRAM & TRIGRAM MODELS

unigram_counts = Counter()
bigram_counts = Counter()
trigram_counts = Counter()

for tokens in tokenized:
    for i, word in enumerate(tokens):
        unigram_counts[word] += 1
        
        if i > 0:
            bigram_counts[(tokens[i - 1], word)] += 1
        
        if i > 1:
            trigram_counts[(tokens[i - 2], tokens[i - 1], word)] += 1

# Probability calculation

bigram_probs = {
    (w1, w2): count / unigram_counts[w1]
    for (w1, w2), count in bigram_counts.items()
}

trigram_probs = {
    (w1, w2, w3): count / bigram_counts[(w1, w2)]
    for (w1, w2, w3), count in trigram_counts.items()
}


# STREAMLIT UI

st.title("🔮 AI-Based Next Word Prediction (Bigram & Trigram Model)")
st.write("This application predicts the next word based on a custom-trained language model.")

# Sidebar 
st.sidebar.markdown('<div class="student-box">🪪 Student Details', unsafe_allow_html=True)
st.sidebar.markdown('<div class="student-title"></div>', unsafe_allow_html=True)

st.sidebar.markdown("""
**Name:** *Anmol Ratan*  
**Reg. No:** *2447210*
""")

st.sidebar.markdown('</div>', unsafe_allow_html=True)  

st.sidebar.markdown('<div class="separator"></div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="student-box">⚙️ Model Settings', unsafe_allow_html=True)

model_choice = st.sidebar.radio("Choose Language Model:", ["Bigram", "Trigram"])
top_k = st.sidebar.slider("Number of Suggestions", 1, 3, 3)

if st.sidebar.checkbox("Show Corpus Statistics"):
    st.sidebar.write(f"📌 Sentences: {len(raw_sentences)}")
    st.sidebar.write(f"📌 Vocabulary Size: {len(unigram_counts)}")
    st.sidebar.write(f"📌 Total Words: {sum(unigram_counts.values())}")



user_input = st.text_input("✍️ Type a word or phrase:")


# PREDICTION FUNCTION

def predict_next(text, top_n=3):
    tokens = preprocess(text)
    if not tokens:
        return []

    if model_choice == "Bigram":
        last = tokens[-1]
        candidates = [(w2, prob) for (w1, w2), prob in bigram_probs.items() if w1 == last]

    else:  # Trigram
        if len(tokens) < 2:
            return ["Enter 2+ words for trigram mode."]
        last_two = tuple(tokens[-2:])
        candidates = [(w3, prob) for (w1, w2, w3), prob in trigram_probs.items() if (w1, w2) == last_two]

    return sorted(candidates, key=lambda x: x[1], reverse=True)[:top_n]


# DISPLAY RESULTS

if user_input:
    results = predict_next(user_input, top_k)
    
    if results:
        st.subheader("🔗 Suggested Next Words")
        df = pd.DataFrame(results, columns=["Word", "Probability"])
        df["Probability"] = df["Probability"].round(4)
        st.table(df)
    else:
        st.warning("⚠️ No predictions found for that phrase.")


# To Show Model Data
with st.expander("📊 View Model Data"):
    st.write("Bigram Count Table")
    st.dataframe(pd.DataFrame(list(bigram_probs.items()), columns=["Bigram", "Probability"]))

    st.write("Trigram Count Table")
    st.dataframe(pd.DataFrame(list(trigram_probs.items()), columns=["Trigram", "Probability"]))
