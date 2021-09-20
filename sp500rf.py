
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from rf_functions import *


################################################
# 1. DATA PREPROCESSING
################################################

# get the data and add features
symbols, data = get_stock_data()
trainX, testX, trainY, testY = add_features(symbols, data)


# scale X if correlated with close val to make stationary
trainX, cors1, cors2 = scale_vals_corr(trainX)
testX, _, _ = scale_vals_corr(testX, cors1, cors2)
trainX = trainX.drop(['Close', 'Ticker', 'Adj Close', 'Volume'], axis=1)
testX = testX.drop(['Close', 'Ticker', 'Adj Close', 'Volume'], axis=1)


# autoscale X, use the train values to generate the scalers
trainXscaled, xscaler = scale_vals_auto(trainX)
# use the scalers from the training set to scale the test set
testXscaled, _ = scale_vals_auto(testX, xscaler)


# transform Y values to boolean
# trainY, testY = target_zscore(trainY, testY)
trainY, testY = target_up_dn(trainY, testY)


################################################
# 2. MACHINE LEARNING
################################################

regr = RandomForestClassifier(n_jobs=6)
print("fitting...")
regr.fit(trainXscaled, trainY)
print("predicting...")
predY = regr.predict(testXscaled)


################################################
# 3. EVALUATE THE MODEL
################################################

# confusion matrix
print( metrics.confusion_matrix(testY, predY) )
y_actual = pd.Series(testY, name='Actual')
y_predicted = pd.Series(predY, name='Predicted', index=testY.index)
print(pd.crosstab(y_actual, y_predicted))


#print accuracy of model
print('Accuracy: '+str(metrics.accuracy_score(testY, predY)))
#print precision value of model
prec=metrics.precision_score(testY, predY)
print('Precision: '+str(prec))
#print recall value of model
recl = metrics.recall_score(testY, predY)
print('Recall: '+str(recl))
#print F1 score of model
print('F1 score: '+str( 2*((prec*recl)/(prec+recl)) ))


# sort feature importance
importance = pd.DataFrame({
    "names":testXscaled.columns.to_list(),
    "values": regr.feature_importances_.tolist()
})
importance = importance.sort_values('values', ascending=False)

# plot feature importance
plt.bar(range(len(importance)), importance['values'])
plt.xticks(range(len(importance)), importance['names'], rotation='vertical')
plt.show()


print('done')