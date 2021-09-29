
import pandas as pd
dataread = pd.read_csv('features.csv',header=None)
import numpy as np
dataread = dataread.drop([0])
flag = ["81920","81921","81922","81923","81924","81925","81926","81927","81928","81929"]
X=dataread.loc[:, ~dataread.columns.isin(flag)].copy()
X = X.to_numpy()
y = dataread.drop(dataread.ix[:, '0':'81919'].columns, axis = 1)
y = np.asarray(y.to_numpy(), dtype=np.int)
from numpy import mean
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.optimizers import Nadam
from keras.optimizers import Adadelta
from keras_adabound import AdaBound

# get the model
def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(n_outputs,input_dim=n_inputs, activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy')
    return model

# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
    results = list()
    resultspre = list()
    resultsrec = list()
    resultsf1 = list()
    resultsh = list()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    cv = RepeatedKFold(n_splits=3, n_repeats=5, random_state=1)
    for train_ix, test_ix in cv.split(X):
        # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        # define model
        model = get_model(n_inputs, n_outputs)
        # fit model
        model.fit(X_train, y_train, verbose=0, epochs=40)
        # make a prediction on the test set
        yhat = model.predict(X_test)
        # round probabilities to class labels
        yhat = yhat.round()
        # calculate accuracy
        acc = accuracy_score(y_test, yhat)
        #calculate precision
        pre = precision_score(y_test, yhat, average='micro')
        #calculate recall
        rec = recall_score(y_test, yhat, average='micro')
        #calculate f1score
        f1 = f1_score(y_test, yhat, average='micro')
        #calculate hamming loss
        h1 = hamming_loss(y_test, yhat)
        # store result
        print('>%.3f' % acc)
        results.append(acc)
        # store result
        print('>%.3f' % pre)
        resultspre.append(pre)
        # store result
        print('>%.3f' % rec)
        resultsrec.append(rec)
        # store result
        print('>%.3f' % f1)
        resultsf1.append(f1)
        # store result
        print('>%.3f' % h1)
        resultsh.append(h1)
        #
        print(classification_report(y_test,yhat))
        #
        print(multilabel_confusion_matrix(y_test, yhat))
    return results,resultspre,resultsrec,resultsf1, resultsh

# evaluate model
results,resultspre,resultsrec,resultsf1,resultsh = evaluate_model(X, y)
# summarize performance
print('Accuracy: %.3f (%.3f)' % (mean(results), std(results)))
print('Precision: %.3f (%.3f)' % (mean(resultspre), std(resultspre)))
print('Recall: %.3f (%.3f)' % (mean(resultsrec), std(resultsrec)))
print('F1: %.3f (%.3f)' % (mean(resultsf1), std(resultsf1)))
print('Hamming Loss: %.3f (%.3f)' % (mean(resultsh), std(resultsh)))




