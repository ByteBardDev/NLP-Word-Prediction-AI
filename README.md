streamlit
pandas
numpy
regex# 🔮 Next Word Prediction Using N-Gram Language Model

This project is a **Next-Word Prediction Application** built using **Python, NLP preprocessing, and a custom-trained Bigram & Trigram model**.  
The model predicts the next most likely word(s) based on user input using conditional probabilities derived from the dataset.

This project includes a fully interactive **Streamlit web application** with an enhanced UI, animated sidebar design, and live prediction results.

---

## 🚀 Features

✔ Trained on a custom text corpus  
✔ Bigram & Trigram probability models  
✔ Text preprocessing (lowercase, stopwords removal, cleaning, tokenization)  
✔ Top-k next-word prediction  
✔ Interactive UI with input box and table display  
✔ Modern animated sidebar with student details  
✔ Built with NLP fundamentals (no deep learning required)

---

## 🧠 How It Works

The model uses statistical probabilities from the corpus:

### 📌 Bigram Probability

\[
P(w_i | w_{i-1}) = \frac{count(w_{i-1}, w_i)}{count(w_{i-1})}
\]

### 📌 Trigram Probability

\[
P(w_i | (w_{i−2}, w_{i−1})) = \frac{count(w_{i−2}, w_{i−1}, w_i)}{count(w_{i−2}, w_{i−1})}
\]

These probabilities are used to suggest the most likely next words.

---

## 📂 Project Structure


