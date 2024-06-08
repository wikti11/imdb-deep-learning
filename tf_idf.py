from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('IMDB Dataset.csv')
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

model_tfidf_nb = make_pipeline(TfidfVectorizer(), MultinomialNB())

model_tfidf_nb.fit(train_data['review'], train_data['sentiment'])

pred_tfidf_nb = model_tfidf_nb.predict(test_data['review'])

accuracy_tfidf_nb = accuracy_score(test_data['sentiment'], pred_tfidf_nb)
print(f'TFIDF + Naive Bayes Accuracy: {accuracy_tfidf_nb}')
print(f'TFIDF + Naive Bayes Classification Report:')
print(classification_report(test_data['sentiment'], pred_tfidf_nb))