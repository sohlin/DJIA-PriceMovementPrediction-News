# DJIA-PriceMovementPrediction-News
Course project for Data-driven Modeling and Analysis

The project use news data to predict DJIA short-term price movement. Use CountVectorizer (sklearn) to transform news to vectors, and this method is One-hot Representation. At the same time, use Word2vec to complete the transformation and compare the results.  Typically, BOW, though simple, performs well in such task. 

Then conduct the prediction through models like LSTM, SVM, and Random Forest and evaluate the results.
***
## Data resource:
Kaggle.com

https://www.kaggle.com/aaron7sun/stocknews
***
## Workflow:
Preprocessing: Remove non-letters, Convert words to lower case and split them, Optionally remove stop words

Words to Vectors: BOW (CountVectorizer), Word2vec

Modeling : Using the vectors as features and the price movement labels as the response variable

Prediction
***
## Results (part)
DJIA Price Plot & train-test-split

![image](https://github.com/sohlin/DJIA-PriceMovementPrediction-News/blob/master/image/pic2.PNG)

Backtest on investment strategy
