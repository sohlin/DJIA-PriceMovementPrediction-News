# DJIA-PriceMovementPrediction-News
*Forecast price movement by LSTM, SVM, and Random Forest using news data*

The project use news data to predict DJIA short-term price movement. Use CountVectorizer (sklearn) to transform news to vectors, and this method is One-hot Representation. At the same time, use Word2vec to complete the transformation and compare the results.  Typically, BOW, though simple, performs well in such task. 

Then conduct the prediction through models like LSTM, SVM, and Random Forest and evaluate the results. Typically, I use news of today to predict the next day's DJIA movement.
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
### DJIA Price Plot & train-test-split

For training and testing, I split the data set into two part. News before 01/02/2015 are set to be training set, while others are test set. It is approximately 8:2 for trainSet:testSet.

At the mean while, The number of rises and falls of DJIA for training is about half to half, so there is no serious sample imbalance problem here.

![image](https://github.com/sohlin/DJIA-PriceMovementPrediction-News/blob/master/image/pic2.PNG)

### Accuracy of test (SVM)

SVM(rbf)|precision|recall|f1-score|support
---|:--:|:--:|:--:|---:
up|1|0.7|0.83|186
down|0.78|1|0.87|191
accuracy|   |		|0.85|377
macro avg|0.89|0.85|0.85|377
weighted avg|0.89|0.85|0.85|377

### Backtest on investment strategy

![image](https://github.com/sohlin/DJIA-PriceMovementPrediction-News/blob/master/image/pic1.PNG)

The net value curve is shown above. In real world, we have to consider handling fee, so I set as below:

Buy commision = 0.48%, Sell commision = 0.5%. 

The back-test details are shown below (SVM):

tradeNumber_long|winRatio_long|cumReturn|annualReturn|excessCumReturn_Open|excessAnnualReturn_Open|maxdown|calmarRatio|sharpeRatio
---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---:
67|52.24%|30.44%|18.85%|29.86%|18.47%|-8.97%|2.10|1.39

