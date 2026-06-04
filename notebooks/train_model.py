import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

df = pd.read_csv("spam.csv", encoding="latin-1")

df = df[['v1','v2']]
df.columns = ['label','message']

df['label'] = df['label'].map({
    'ham':0,
    'spam':1
})

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df['message'])
y = df['label']

model = MultinomialNB()
model.fit(X,y)

pickle.dump(model,open('model.pkl','wb'))
pickle.dump(vectorizer,open('vectorizer.pkl','wb'))

print("Model Trained Successfully")