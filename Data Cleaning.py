"""
# Parse Data Manually
"""
# Read in the raw text

rawData = open("SMSSpamCollection.tsv").read()

# Print the raw data
rawData[0:500]

# order data in df
parsedData = rawData.replace('\t', '\n').split('\n')
labelList = parsedData[0::2]
textList = parsedData[1::2]

import pandas as pd

fullCorpus = pd.DataFrame({
    'label': labelList[:-1],
    'body_list': textList
})

fullCorpus.head()

"""
# Parse Data Automate
"""
fullCorpus = pd.read_csv("SMSSpamCollection.tsv", sep="\t", header=None)
fullCorpus.columns = ['label', 'body_text']
fullCorpus.head()

"""
Analyze Data
"""
# How many spam/ham are there?
print("Out of {} rows, {} are spam, {} are ham".format(len(fullCorpus),
                                                       len(fullCorpus[fullCorpus['label']=='spam']),
                                                       len(fullCorpus[fullCorpus['label']=='ham'])))
# How much missing data is there?

print("Number of null in label: {}".format(fullCorpus['label'].isnull().sum()))
print("Number of null in text: {}".format(fullCorpus['body_text'].isnull().sum()))

"""
Remove punctuation:
# punctuation are: 
         !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
"""
import string

data = pd.read_csv("SMSSpamCollection.tsv", sep='\t', header=None)
data.columns = ['label', 'body_text']

def remove_punct(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    return text_nopunct

data['body_text_clean'] = data['body_text'].apply(lambda x: remove_punct(x))

data.head()

"""
Tokenize Data:
tokenizing it's basically make a list of words.
We will use the '\W+' regex which take every non-word char as separator.
"""
import re

def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens

data['body_text_tokenized'] = data['body_text_clean'].apply(lambda x: tokenize(x.lower()))

data.head()

"""
Remove Stopwords.
# Stopwords are non important words. like:
        is, I, have
"""
import nltk
nltk.download('stopwords')

stopword = nltk.corpus.stopwords.words('english')

def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]
    return text

data['body_text_nostop'] = data['body_text_tokenized'].apply(lambda x: remove_stopwords(x))

data.head()

"""
Stemming:
take every word and cut it to find basic root. (only cut)
"""
ps = nltk.PorterStemmer()

def stemming(tokenized_text):
    text = [ps.stem(word) for word in tokenized_text]
    return text

data['body_text_stemmed'] = data['body_text_nostop'].apply(lambda x: stemming(x))

data.head()

"""
Lemmatizing:
find basic real root for every word. (Lemma)
"""
wn = nltk.WordNetLemmatizer()

def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

data['body_text_lemmatized'] = data['body_text_nostop'].apply(lambda x: lemmatizing(x))

data.head(10)

