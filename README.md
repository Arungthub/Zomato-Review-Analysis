# Zomato Review Sentiment Analysis Project

This project performs **sentiment analysis** on Zomato customer reviews using **Natural Language Processing (NLP)** and **Machine Learning**.
The goal is to classify reviews as **positive (1)** or **negative (0)** using text processing techniques and a **Naïve Bayes classifier**.

---

#  Project Overview

Customer reviews play a major role in understanding user satisfaction.
This project:

* Loads and preprocesses customer reviews
* Cleans text and removes unnecessary words
* Converts text into numerical features using **Bag-of-Words (CountVectorizer)**
* Trains a **Gaussian Naïve Bayes classifier**
* Evaluates the model with accuracy and confusion matrix

This workflow is commonly used in real-world sentiment analysis tasks such as product reviews, food delivery apps, and customer feedback forms.

---

# 📂 **Project Structure**

```
Zomato-Review-Analysis/
│
├── Zomato Project/
│   ├── Zomato.py              # Main sentiment analysis script
│   ├── Zomato.csv             # Dataset with reviews + labels
│   ├── requirements.txt        # Dependencies
│
└── README.md                   # Project documentation
```

---

# Dataset Information

The dataset contains two columns:

| Column     | Description                              |
| ---------- | ---------------------------------------- |
|   Review   | Customer review text                     |
|   Liked    | 1 = Positive review, 0 = Negative review |

---

# Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* NLTK (Natural Language Toolkit)
* Scikit-learn (Machine Learning)

---

# Model Used

# Gaussian Naive Baye

Naïve Bayes is:

* Fast
* Simple
* Works well for text classification
* Handles high-dimensional data effectively

Ideal for sentiment analysis involving large vocabulary sizes.

# How to Run the Project

#1. Clone the Repository

--> git clone <your-repo-link> / create a codespace and execute the code.
--> cd Zomato-Review-Analysis


# 2. Go to Project Folder

-->cd "Zomato Project"


# 3. Install Required Libraries
-->pip install -r requirements.txt

# 4. Run Script

--> python Zomato.py


# 🧹 **NLP Preprocessing Steps**

The reviews undergo the following cleaning steps:

1. Removing non-alphabet characters
2. Converting to lowercase
3. Tokenization (splitting into words)
4. Removing stopwords (except **not**)
5. Stemming using **PorterStemmer**
6. Building corpus
7. Converting to numeric features using **CountVectorizer**

---

# Model Evaluation

The project computes:

Confusion Matrix
Accuracy Score


# Key Learning Outcomes

By completing this project, you learn:

* How to preprocess text for NLP
* How to build a Bag-of-Words model
* How to apply Naïve Bayes classifier
* How to evaluate classification performance
* How to structure a complete NLP pipeline

---

# Future Improvements

You can extend this project by adding:

* TF-IDF Vectorizer
* Logistic Regression or SVM
* WordCloud visualizations
* Deep learning models (LSTM, BERT)
* Deployment using Flask/Streamlit


