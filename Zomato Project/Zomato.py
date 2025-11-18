#import files

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load Dataset

dataset = pd.read_csv(r"/workspaces/Zomato-Review-Analysis/Zomato Project/Zomato.csv", encoding='latin-1')


dataset.columns = ['Review', 'Liked']

#Data Process

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
if 'not' in all_stopwords:
    all_stopwords.remove('not')

for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

#  Use CountVectorizer after building corpus

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()

# Correct labels

y = dataset['Liked'].astype(int).values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Prediction Classifier

y_pred = classifier.predict(X_test)

# confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Accuraccy of the model 

print("Accuracy:", accuracy_score(y_test, y_pred))
