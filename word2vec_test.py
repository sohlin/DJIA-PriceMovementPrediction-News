#--coding:utf-8--

import pandas as pd
import numpy as np
from gensim.models import word2vec
import re
import time


def news_to_words(news):
    '''
    Function to convert news title to strings of words.
    The input is a Dataframe (news title), and
    the output is an array (preprocessed news).
    Based on scikit-learn the Bag of Words model
    :param news:
    :return: headlines
    '''
    # Removing punctuations
    news.replace(to_replace="[^a-zA-Z]", value=" ", regex=True, inplace=True)

    # Renaming column names for ease of access
    list = [i for i in range(len(news.columns))]
    newIndex = [str(i) for i in list]
    news.columns = newIndex

    # Convertng headlines to lower case
    for index in newIndex:
        news[index] = news[index].str.lower()

    sentences = []
    for row in range(0, len(news.index)):
        for x in news.iloc[row, 0:len(news.columns)]:
            sentences.append(str(x).strip().lstrip('b').split())

    return sentences


start = time.time()

data = pd.read_csv('Combined_News_DJIA.csv', encoding="ISO-8859-1")
stockPrediction = data['Label'].shift(-1)
data['PriceMovement'] = stockPrediction
data = data[np.isfinite(data['PriceMovement'])]

train = data[data['Date'] < '20150102']
test = data[data['Date'] > '20141231']

# Get news for train
newsData = train.iloc[:, 2:(len(data.columns)-1)]
trainNews = news_to_words(newsData)


# Word vector dimensionality
num_featrues = 300
# Minimum word count
min_word_count = 10
# Number of threads to run in parallel
num_workers = 2
# Context window size
context = 5
# Downsample setting for frequent words
downsampling = 1e-3

model = word2vec.Word2Vec(trainNews, workers=num_workers, \
                          size=num_featrues, min_count=min_word_count, window=context, sample=downsampling)

model.init_sims(replace=True)

model_name = "300features_10minWords_5winSize"
model.save(model_name)
# model.wv.save_word2vec_format("data/model/word2vec_org","data/model/vocabulary",binary=False)