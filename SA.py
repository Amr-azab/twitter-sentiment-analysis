import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import numpy as np
import string


df = pd.read_csv('Twitter_Data.csv')

#print(f"Number of rows: {df.shape[0]}")
#print(f"Number of columns: {df.shape[1]}")
#print(df.head())

#stopwords.words('english')

punctuation_list = list(string.punctuation)
def preprocess_text(text):
    if pd.isnull(text):
        return ""
    else:
        text = text.lower()
        text = ''.join(c for c in text if c.isalnum() or c.isspace()) 
        tokens = word_tokenize(text)
        
        negation = False
        result = []
        for token in tokens:
            if negation:
                result.append("NOT_" + token)
                negation = False
            elif token in ["not", "never", "no"]:
                negation = True
                result.append(token)
            else:
                result.append(token)

        filtered_tokens = [token for token in result if token not in punctuation_list]
        preprocessed_text = ' '.join(filtered_tokens)
        return preprocessed_text
    
df['clean_text'] = df['clean_text'].apply(preprocess_text)

df.dropna(subset=['clean_text', 'category'], inplace=True)


# print(df[['clean_text', 'category']].head())


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['category']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred, average='weighted') * 100
recall = recall_score(y_test, y_pred, average='weighted') * 100
f1 = f1_score(y_test, y_pred, average='weighted') * 100
print ('######################### Evaluating the model  ##################')
print('Accuracy:', accuracy, '%')
print('Precision:', precision, '%')
print('Recall:', recall, '%')
print('F1 Score:', f1, '%')

print ('############################ Enter the number of texts ########################')
num_new_texts = int(input("How many texts do you want to classify? "))
new_texts = []
for i in range(num_new_texts):
    new_text = input(f"Enter text {i+1}: ")
    new_texts.append(new_text)


preprocessed_new_texts = [preprocess_text(text) for text in new_texts]

X_new = vectorizer.transform(preprocessed_new_texts)

y_new_pred = clf.predict(X_new)

sentiment_mapping = {1.0: 'positive', -1.0: 'negative', 0.0: 'natural'}

print('############### The result #########################')
for i in range(len(new_texts)):
    print(f"Text: '{new_texts[i]}'")
    predicted_sentiment = sentiment_mapping[y_new_pred[i]]
    print(f"Predicted sentiment: '{predicted_sentiment}'")
    print("------")


