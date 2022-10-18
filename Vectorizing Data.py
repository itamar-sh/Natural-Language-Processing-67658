"""
get data():
get and clean the data
"""
import pandas as pd
import re
import string
import nltk

def get_data():
    pd.set_option('display.max_colwidth', 100)
    data = pd.read_csv("SMSSpamCollection.tsv", sep='\t')
    data.columns = ['label', 'body_text']
    data['cleaned_text'] = data['body_text'].apply(lambda x: " ".join(clean_text(x)))
    return data

def clean_text(text):
    ps = nltk.PorterStemmer()
    stopwords = nltk.corpus.stopwords.words('english')

    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

"""
1) Count vectorization:
Creates a document-term matrix where the entry of each cell will be 
a count of the number of times that word occurred in that document.
"""

data = get_data()
data = data[0:20]  # only for display

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(analyzer=clean_text)
X_counts = count_vect.fit_transform(data['body_text'])
print("Count vectorization")
print(X_counts.shape)
print(count_vect.get_feature_names())
print(pd.DataFrame(X_counts.toarray()))

"""
2) Vectorizing Raw Data: N-Grams

Creates a document-term matrix where counts still occupy the cell 
but instead of the columns representing single terms, 
they represent all combinations of adjacent words of length n in your text.

"NLP is an interesting topic"

| n | Name      | Tokens                                                         |
|---|-----------|----------------------------------------------------------------|
| 2 | bigram    | ["nlp is", "is an", "an interesting", "interesting topic"]      |
| 3 | trigram   | ["nlp is an", "is an interesting", "an interesting topic"] |
| 4 | four-gram | ["nlp is an interesting", "is an interesting topic"]    |
"""

from sklearn.feature_extraction.text import CountVectorizer

ngram_vect = CountVectorizer(ngram_range=(2,2))
X_counts = ngram_vect.fit_transform(data['cleaned_text'])
print("N-Grams Count vectorization")
print(X_counts.shape)
print(ngram_vect.get_feature_names())
print(pd.DataFrame(X_counts.toarray()))

"""
3) Vectorizing Raw Data: TF-IDF

Creates a document-term matrix where the columns represent single unique terms (unigrams) 
but the cell represents a weighting meant to represent 
how important a word is to a document.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(analyzer=clean_text)
X_tfidf = tfidf_vect.fit_transform(data['body_text'])
print(X_tfidf.shape)
print(tfidf_vect.get_feature_names())
print(pd.DataFrame(X_counts.toarray()))