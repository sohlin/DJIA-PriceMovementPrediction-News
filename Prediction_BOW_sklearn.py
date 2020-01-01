#--coding:utf-8--

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')


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

    headlines = []
    for row in range(0, len(news.index)):
        headlines.append(' '.join(str(x) for x in news.iloc[row, 0:len(news.columns)]))

    return headlines

if __name__ == '__main__':
    # Data from Aaron7sun, Kaggle.com
    data = pd.read_csv('Combined_News_DJIA.csv', encoding="ISO-8859-1")
    stockPrediction = data['Label'].shift(-1)
    data['PriceMovement'] = stockPrediction
    data = data[np.isfinite(data['PriceMovement'])]

    train = data[data['Date'] < '20150102']
    test = data[data['Date'] > '20141231']

    # Get news for train & test
    newsData = train.iloc[:, 2:(len(data.columns)-1)]
    trainNews = news_to_words(newsData)
    testNews = []
    for row in range(0, len(test.index)):
        testNews.append(' '.join(str(x) for x in test.iloc[row, 2:(len(data.columns)-1)]))

    # Perform word to vector
    simpleVectorizer = CountVectorizer(stop_words='english',ngram_range=(1, 1))
    transformer = TfidfTransformer()
    x_train = transformer.fit_transform(simpleVectorizer.fit_transform(trainNews))
    x_test = transformer.transform(simpleVectorizer.transform(testNews))

    # Logistic Regression
    logreg = LogisticRegression(C=0.1)
    logreg = logreg.fit(x_train, train["PriceMovement"])
    predictionsLogreg = logreg.predict(x_test)
    testResults_logreg = pd.crosstab(test['PriceMovement'], predictionsLogreg, rownames=['Actual'], colnames=['Predicted'])
    print("Train score:" + str(logreg.score(x_train, train['PriceMovement'])))
    print(classification_report(test['PriceMovement'], predictionsLogreg))
    print(accuracy_score(test['PriceMovement'], predictionsLogreg))

    # Support Vector Machines (SVM) with rbf kernel
    svc = SVC(C=1, class_weight='balanced', kernel='rbf', gamma=0.1, tol=1e-10)
    svc = svc.fit(x_train, train['PriceMovement'])
    predictionsSvc = svc.predict(x_test)
    testResults_svc = pd.crosstab(test['PriceMovement'], predictionsSvc, rownames=['Actual'], colnames=['Predicted'])
    print("Train score:"+str(svc.score(x_train, train['PriceMovement'])))
    print(classification_report(test['PriceMovement'], predictionsSvc))
    print(accuracy_score(test['PriceMovement'], predictionsSvc))

    # Naive Bayes
    bayes = GaussianNB()
    bayes = bayes.fit(x_train.toarray(), train['PriceMovement'])
    predictionsBayes = bayes.predict(x_test.toarray())
    testResults_bayes = pd.crosstab(test['PriceMovement'], predictionsBayes, rownames=['Actual'], colnames=['Predicted'])
    print("Train score:" + str(bayes.score(x_train.toarray(), train['PriceMovement'])))
    print(classification_report(test['PriceMovement'], predictionsBayes))
    print(accuracy_score(test['PriceMovement'], predictionsBayes))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, criterion='entropy', max_features='auto')
    rf = rf.fit(x_train, train['PriceMovement'])
    predictionsRf = rf.predict(x_test)
    testResults_rf = pd.crosstab(test['PriceMovement'], predictionsRf, rownames=['Actual'], colnames=['Predicted'])
    print("Train score:" + str(rf.score(x_train, train['PriceMovement'])))
    print(classification_report(test['PriceMovement'], predictionsRf))
    print(accuracy_score(test['PriceMovement'], predictionsRf))

    # Output the result and prediction
    dataOutput = pd.concat([testResults_logreg, testResults_svc, testResults_bayes, testResults_rf],axis=0)
    PredictionOutput = pd.DataFrame(index = test.index)
    PredictionOutput['LogisticRegression'] = predictionsLogreg
    PredictionOutput['SVC'] = predictionsSvc
    PredictionOutput['NaiveBayes'] = predictionsBayes
    PredictionOutput['RandomForest'] = predictionsRf
    dataOutput.to_excel("DJIA_ClassificationResult.xlsx")
    PredictionOutput.to_excel("DJIA_prediction_4m.xlsx")









