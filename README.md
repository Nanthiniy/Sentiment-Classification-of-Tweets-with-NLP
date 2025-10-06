# Sentiment Classification of Tweets with NLP

This project performs **sentiment analysis on Twitter data** using **BERT (Bidirectional Encoder Representations from Transformers)**. It classifies tweets as **Positive**, **Negative**, or **Neutral**, leveraging state-of-the-art NLP techniques for high accuracy.

---

## 📘 Overview

Social media platforms like Twitter generate vast amounts of data daily. Understanding users’ emotions can provide valuable insights for businesses, politics, and social research.
This project fine-tunes a pre-trained BERT model to analyze sentiments in tweets effectively.

---

## 🚀 Features

* Preprocessing and cleaning of raw tweet text
* Tokenization using BERT tokenizer
* Fine-tuning of pre-trained BERT model on sentiment dataset
* Model evaluation with accuracy and classification report
* Visualizations of sentiment distribution

---

## 🧠 Tech Stack

* **Python 3**
* **TensorFlow / PyTorch**
* **Transformers (Hugging Face)**
* **Pandas, NumPy**
* **Matplotlib / Seaborn**
* **Scikit-learn**

---

## 📁 Project Structure

```
📦 twitter-sentiment-analysis-BERT
 ┣ 📜 twitter sentiment analysis using BERT.ipynb
 ┣ 📜 README.md
 ┗ 📜 requirements.txt (optional)
```

---

## 🧩 How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/twitter-sentiment-analysis-BERT.git
   ```
2. **Navigate into the folder**

   ```bash
   cd twitter-sentiment-analysis-BERT
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
4. **Run the notebook**

   * Open `twitter sentiment analysis using BERT.ipynb` in **Jupyter Notebook** or **Google Colab**.
   * Execute all cells to train/evaluate the model.

---

## 📊 Results

* Achieved high accuracy using fine-tuned BERT.
* Model effectively distinguishes between positive, negative, and neutral sentiments in tweets.
* Example output visualization:

| Sentiment | Example Tweet               |
| --------- | --------------------------- |
| Positive  | “I love this movie!”        |
| Negative  | “This was a waste of time.” |
| Neutral   | “The event is tomorrow.”    |

<img width="1210" height="638" alt="Screenshot 2025-10-07 004656" src="https://github.com/user-attachments/assets/5bd9953f-abd9-4a90-810f-8c79c6117edf" />

<img width="1283" height="673" alt="Screenshot 2025-10-07 004642" src="https://github.com/user-attachments/assets/cf7ce6b0-13e6-4afc-8358-be2aa3a16ca7" />

---

## 🧾 Future Improvements

* Deploy model using **Flask** or **Streamlit**
* Integrate real-time tweet fetching via **Twitter API**
* Expand dataset for multilingual sentiment analysis

---


