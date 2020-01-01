#--coding:utf-8--

import pandas as pd
import numpy as np
from gensim.models import word2vec
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

import pdb


def news_to_words(news, removeStopwords = False):
    '''
    Function to convert news title to strings of words.
    The input is a Dataframe (news title), and
    the output is an array (preprocessed news).
    Based on scikit-learn the Bag of Words model
    :param news: a Dataframe of news
    :return: headlines: word list for everyday news
    '''
    # Removing punctuations
    news.replace(to_replace="[^a-zA-Z]", value=" ", regex=True, inplace=True)

    # Renaming column names for ease of access
    list = [i for i in range(len(news.columns))]
    newIndex = [str(i) for i in list]
    news.columns = newIndex

    # Show all the stopwords
    stops = set(stopwords.words("english"))

    # Convertng headlines to lower case
    for index in newIndex:
        news[index] = news[index].str.lower()

    sentences = []
    for row in range(0, len(news.index)):
        headline = []
        for x in news.iloc[row, 0:len(news.columns)]:
            headline = headline + str(x).strip().lstrip('b').split()

        if removeStopwords:
            headline = [w for w in headline if not w in stops]

        sentences.append(headline)

    return sentences


def makeFeatureVec(words, model, num_features):
    '''
    # Function to average all of the word vectors in a given paragraph
    :param words: a list of words
    :param model: the word2vec model (after training)
    :param num_features: the dimension of each word vector
    :return: a vector of the average of all of the word vectors in a given paragraph
    '''

    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)

    # Loop over each word in the news and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])

    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)

    return featureVec


def getAvgFeatureVecs(news, model, num_features):
    '''
    Given a set of news (each one a list of words), calculate
    the average feature vector for each one and return a 2D numpy array.
    Initialize a counter
    :param news: word list
    :param model: the word2vec model (after training)
    :param num_features: the dimension of each word vector
    :return: everyday news vector (2D array)
    '''
    counter = 0

    # Preallocate a 2D numpy array, for speed
    newsFeatureVecs = np.zeros((len(news),num_features),dtype="float32")
    # Loop through the reviews
    for row in news:
       # Call the function (defined above) that makes average feature vectors
       newsFeatureVecs[counter] = makeFeatureVec(row, model, num_features)
       # Increment the counter
       counter = counter + 1

    return newsFeatureVecs


def k_means_words(model, number):
    '''
    # Function to return the clustering centroid words map using Kmeans.
    :param model: the word2vec model (after training)
    :param number: the number of words in each centroid
    :return: a dict of word centroid
    '''
    wordVectors = model.wv.syn0
    numClusters = int(wordVectors.shape[0]/number)

    clustering = KMeans(n_clusters = numClusters)
    idx = clustering.fit_predict(wordVectors)

    wordCentroidMap = dict(zip(model.wv.index2word, idx))

    return wordCentroidMap,numClusters


def create_bag_of_centroids(wordList, wordCentroidMap):
    '''
    # The number of clusters is equal to the highest cluster index in the word / centroid map
    :param wordList: The everyday news list ( Split by words)
    :param wordCentroidMap:
    :return: Return the "bag of centroids"
    '''
    num_centroids = max( wordCentroidMap.values() ) + 1

    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )

    # Loop over the words in the news. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count by one
    for word in wordList:
        if word in wordCentroidMap:
            index = wordCentroidMap[word]
            bag_of_centroids[index] += 1

    return bag_of_centroids



if __name__ == '__main__':

    model = word2vec.Word2Vec.load("300features_10minwords")
    #type(model.wv.syn0)
    #model.wv.syn0.shape

    data = pd.read_csv('Combined_News_DJIA.csv', encoding="ISO-8859-1")
    stockPrediction = data['Label'].shift(-1)
    data['PriceMovement'] = stockPrediction
    data = data[np.isfinite(data['PriceMovement'])]

    train = data[data['Date'] < '20150102']
    test = data[data['Date'] > '20141231']
    test = test.reindex()

    # Get news to words for train & test
    trainData = train.iloc[:, 2:(len(data.columns) - 1)]
    trainNews = news_to_words(trainData, removeStopwords=True)
    testData = test.iloc[:, 2:(len(data.columns) - 1)]
    testNews =  news_to_words(testData, removeStopwords=True)

    '''
    # Method 1. Transform words to vectors ( Vector average)
    trainVecs = getAvgFeatureVecs(trainNews, model, num_features=300)
    testVecs = getAvgFeatureVecs(testNews, model, num_features=300)
    '''

    # Method 2. Transform words to vectors ( Clustering)
    numberInCentroid = 10
    wordCentroidMap,numClusters = k_means_words(model, numberInCentroid)

    counter = 0
    trainVecs = np.zeros((len(trainNews), numClusters), dtype="float32")
    for news in trainNews:
        trainVecs[counter] = create_bag_of_centroids(news, wordCentroidMap)
        counter += 1
    counter = 0
    testVecs = np.zeros((len(testNews), numClusters), dtype="float32")
    for news in testNews:
        testVecs[counter] = create_bag_of_centroids(news, wordCentroidMap)
        counter += 1


    # Random Forest Train & Predict
    rf = RandomForestClassifier(n_estimators=300)
    rf = rf.fit(trainVecs, train["PriceMovement"])
    predictionsRf = rf.predict(testVecs)
    testResults_rf = pd.crosstab(test['PriceMovement'], predictionsRf, rownames=['Actual'], colnames=['Predicted'])
    print("Train score:" + str(rf.score(trainVecs, train['PriceMovement'])))
    print(classification_report(test['PriceMovement'], predictionsRf))
    print(accuracy_score(test['PriceMovement'], predictionsRf))

    # SVM rbf Train & Predict
    svc = SVC(C=5, class_weight='balanced', kernel='rbf', gamma=0.1, tol=1e-10)
    svc = svc.fit(trainVecs, train['PriceMovement'])
    predictionsSvc = svc.predict(testVecs)
    testResults_svc = pd.crosstab(test['PriceMovement'], predictionsSvc, rownames=['Actual'], colnames=['Predicted'])
    print("Train score:" + str(svc.score(trainVecs, train['PriceMovement'])))
    print(classification_report(test['PriceMovement'], predictionsSvc))
    print(accuracy_score(test['PriceMovement'], predictionsSvc))











