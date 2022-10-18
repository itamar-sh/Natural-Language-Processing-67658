"""
get data
"""
import inline as inline
import numpy as np
import pandas as pd
# from matplotlib import pyplot
from sympy.physics.control.control_plots import matplotlib
from sympy.physics.quantum.circuitplot import pyplot

data = pd.read_csv("SMSSpamCollection.tsv", sep='\t')
data.columns = ['label', 'body_text']
"""
1) 'body_len' feature
How long the body.
"""
data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(" "))
data.head()

"""
2) 'punct%' feature
How much of the text is punctuations.
"""
import string

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100

data['punct%'] = data['body_text'].apply(lambda x: count_punct(x))
data.head()

"""
evaluate feautures.
body_len is good but punct% is not. 
Because body_len make a separation of the span and ham.
"""
bins = np.linspace(0, 200, 40)

pyplot.hist(data[data['label']=='spam']['body_len'], bins, alpha=0.5, label='spam')
pyplot.hist(data[data['label']=='ham']['body_len'], bins, alpha=0.5, label='ham')
pyplot.legend(loc='upper left')
pyplot.show()

bins = np.linspace(0, 50, 40)

pyplot.hist(data[data['label']=='spam']['punct%'], bins, alpha=0.5, label='spam')
pyplot.hist(data[data['label']=='ham']['punct%'], bins, alpha=0.5, label='ham')
pyplot.legend(loc='upper right')
pyplot.show()

"""
Box-Cox Power Transformation
use exponent on each value to yield distribution.
"""

for i in [1, 2, 3, 4, 5]:
    pyplot.hist((data['punct%'])**(1/i), bins=40)
    pyplot.title("Transformation: 1/{}".format(str(i)))
    pyplot.show()