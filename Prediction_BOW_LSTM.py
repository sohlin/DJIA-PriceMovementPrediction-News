#--coding:utf-8--

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix


def news_to_words(news):
    '''
    Function to convert news title to strings of words.
    The input is a Dataframe (news title), and
    the output is an array (preprocessed news).
    Based on scikit-learn the Bag of Words model
    :param news: The DataFrame of ordinary news data
    :return: headlines: Processed news data (list)
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

    headlines = []
    for row in range(0, len(news.index)):
        headlines.append(' '.join(str(x) for x in news.iloc[row, 0:len(news.columns)]))

    return headlines


def lstm_model(inputShape_0, inputShape_1):
    '''
    Function to build a 5 layer LSTM model based on Keras
    :param inputShape_0: The dimension of each sample, which is the row
    :param inputShape_1: The dimension of each sample, which is the column
    :return: regressor
    '''

    lstm = models.Sequential()
    lstm.add(layers.LSTM(units=50, return_sequences=True, input_shape=(inputShape_0, inputShape_1)))
    lstm.add(layers.Dropout(0.2))
    lstm.add(layers.LSTM(units=50, return_sequences=True))
    lstm.add(layers.Dropout(0.2))
    lstm.add(layers.LSTM(units=50, return_sequences=True))
    lstm.add(layers.Dropout(0.2))
    lstm.add(layers.LSTM(units=50))
    lstm.add(layers.Dropout(0.2))
    lstm.add(layers.Dense(units=1))
    lstm.compile(optimizer='adam', loss='mean_squared_error')

    return lstm


stockData = pd.read_csv("upload_DJIA_table.csv")
stockTrain = stockData[stockData['Date'] < '20150102']
stockTest = stockData[stockData['Date'] > '20141231']

# Adding a datetime index
stockData['datetime'] = pd.to_datetime(stockData['Date'])

# Feature scaling
#scalar = MinMaxScaler(feature_range = (0, 1))
#stockTrain_scaled = scalar.fit_transform(stockTrain.iloc[:,4:5].values)

# Data from Aaron7sun, Kaggle.com
combinedData = pd.read_csv('Combined_News_DJIA.csv', encoding="ISO-8859-1")
stockPrediction = combinedData['Label'].shift(-1)
combinedData['PriceMovement'] = stockPrediction
combinedData = combinedData[np.isfinite(combinedData['PriceMovement'])]

newsTrain = combinedData[combinedData['Date'] < '20150102']
newsTest = combinedData[combinedData['Date'] > '20141231']

# Get news for train & test
newsData = newsTrain.iloc[:, 2:(len(combinedData.columns) - 1)]
trainNews = news_to_words(newsData)
testNews = news_to_words(newsTest.iloc[:, 2:(len(combinedData.columns) - 1)])

# Perform word to vector
simpleVectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 1))
x_train = simpleVectorizer.fit_transform(trainNews)
x_test = simpleVectorizer.transform(testNews)

# Generate input array in steps
X_train = []
y_train = []
x_train_array = x_train.toarray().astype('int32')
for i in range(3, len(stockTrain['Close'])):
   X_train.append(x_train_array[i-3:i, :])
   y_train.append(combinedData['Label'][i])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(X_train[0][0])))

x_test_array = x_test.toarray().astype('int32')
newsTest = newsTest.reset_index()
X_test = []
y_test = []
for i in range(3, len(newsTest)):
   X_test.append(x_test_array[i-3:i, :])
   y_test.append(newsTest['Label'][i])
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(X_test[0][0]))) # 32083

# Fit the model
model = lstm_model(X_train.shape[1], len(X_train[0][0]))
model.fit(X_train, y_train, epochs = 100, batch_size = 32)

'''
# Save & Reload the model after training
regressor = models.load_model('my_model.h5')
regressor.save('my_model.h5')

# Attention: if reloading the model, then the following is needed
X_test = X_test.astype('float32') 
'''

# Predict the stock movement
prediction = model.predict(X_test)
prediction = prediction.astype('int32')

# Evaluate the prediction
print(classification_report(y_test.reshape(-1,1), prediction))
print(accuracy_score(y_test.reshape(-1,1), prediction))

dataOutput = pd.DataFrame(prediction)
dataOutput.to_excel('LSTM_prediction.xlsx')