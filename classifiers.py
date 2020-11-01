#######################################################################
# Classifiers library
#
#   This library contains some classes implementing several classifiers
#   such as SVM, LAR, RF, LDA and a Mix classifier using all of them.
#   It also contains a class to build folds for cross validation in
#   several ways.
#
#
# Author: Saugat Bhattacharyya
#         Brain-Computer Interfaces Lab
#         University of Essex
#
#######################################################################

import sys
import numpy as N
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import svm
from sklearn.linear_model import LassoLars, LogisticRegressionCV, LogisticRegression, \
    LinearRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss, make_scorer
#from keras.models import Sequential
#from keras.layers.core import Dense
#from keras.utils import np_utils
#from keras.layers.advanced_activations import ELU
#from keras.layers.core import Activation, Dropout
#from keras.optimizers import Adam, SGD


class Classifier(object):
    def __init__(self):
        self.cl = None

    def train(self, x, y):
        raise NotImplementedError

    def test(self, x):
        raise NotImplementedError

    def test_prediction(self, x):
        raise NotImplementedError

    def test_confidence(self, x):
        raise NotImplementedError

    def train_and_test(self, training_x, training_y, testing_x):
        self.train(training_x, training_y)
        return self.test(testing_x)


# Implementation of a Support Vector Machine classifier using scikit.learn
class SVMKitClassifier(Classifier):
    def __init__(self, c_value=1e-9, kernel="linear", weight=1.0):
        Classifier.__init__(self)
        self.cl = svm.SVC(kernel=kernel, C=c_value, class_weight={1: weight})

    def train(self, training_x, training_y):
        self.cl.fit(N.asarray(training_x), N.asarray(training_y))

    def test_confidence(self, testing_x):
        return N.asarray(self.cl.decision_function(testing_x))

    def test_prediction(self, testing_x):
        return N.asarray(self.cl.predict(testing_x))

    def optimise_param(self, training_x, training_y):
        tuned_parameters = [{
                                'kernel': ['rbf'],
                                'gamma': [1e-3, 1e-4, 1e-6],
                                'C': [1e-9, 1e-6, 1e-3, 1, 1e3, 1e6]
                            },
                            {
                                'kernel': ['linear'],
                                'C': [1e-9, 1e-6, 1e-3, 1, 1e3, 1e6]
                            }]
        clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=10, scoring='accuracy')
        clf.fit(training_x, training_y)
        print("Best estimator")
        print(clf.best_estimator_)


# Implementation of a Least Angle Regression classifier
class LARKitClassifier(Classifier):
    def __init__(self):
        Classifier.__init__(self)
        self.cl = LassoLars()

    def train(self, training_x, training_y):
        self.cl.fit(training_x, training_y)

    def test(self, testing_x):
        return N.asarray(self.cl.predict(testing_x))


# Implementation of a Random Forest classifier
class RFClassifier(Classifier):
    def __init__(self, estimators=100):
        Classifier.__init__(self)
        self.forest = RandomForestClassifier(n_estimators=estimators, random_state=4)

    def train(self, training_x, training_y):
        self.forest = self.forest.fit(training_x, N.ravel(training_y))

    def test_confidence(self, testing_x):
        return self.forest.predict_proba(testing_x)

    def test_prediction(self, testing_x):
        return self.forest.predict(testing_x)


# Implementation of a Linear Discriminant Analysis classifier
class LDAClassifier(Classifier):
    def __init__(self):
        Classifier.__init__(self)
        self.lda = LDA(solver='lsqr', shrinkage='auto')

    def train(self, training_x, training_y):
        y = N.ravel(N.asarray(training_y))
        x = N.asarray(training_x)
        self.lda.fit(x, y)

    def test_confidence(self, testing_x):
        return self.lda.decision_function(testing_x)

    def test_prediction(self, testing_x):
        return self.lda.predict(testing_x)


class LogisticRegressionClassifier(Classifier):
    def __init__(self):
        Classifier.__init__(self)
        self.cl = LogisticRegression(C=1e3, penalty="l2", n_jobs=-1, random_state=0,
                                     class_weight={-1: 1, 1: 1},
                                     #fit_intercept=False,
                                     #class_weight="balanced",
                                     solver="liblinear")

    def train(self, training_x, training_y):
        y = N.ravel(N.asarray(training_y))
        x = N.asarray(training_x)
        self.cl.fit(x, y)
        # print(self.cl.coef_)

    def test_confidence(self, testing_x):
        # Return the "probability" of the "correct" class (first label)
        # return self.cl.predict_proba(testing_x)[:, 0]
        return N.clip(-self.cl.decision_function(testing_x), 0.00001, N.inf)

    def test_prediction(self, testing_x):
        return N.clip(self.cl.predict(testing_x), 0.00001, 1.0)


class LogisticRegressionPredictClassifier(Classifier):
    def __init__(self):
        Classifier.__init__(self)
        self.cl = LogisticRegression(C=1, penalty="l2", n_jobs=-1, random_state=0,
                                     #class_weight={-1: 1, 1: 1},
                                     #fit_intercept=False,
                                     #max_iter= 1e4,
                                     class_weight="balanced",
                                     multi_class= "ovr",
                                     solver="liblinear")

    def train(self, training_x, training_y):
        y = (N.ravel(N.asarray(training_y))*10).astype('int')
        x = N.asarray(training_x)
        self.cl.fit(x, y)
        # print(self.cl.coef_)

    def test(self, testing_x):
        # Return the "probability" of the "correct" class (first label)
        # return self.cl.predict_proba(testing_x)[:, 0]
        return N.clip(self.cl.predict(testing_x)/10,0.00001,1.0)


class LogisticRegressionCVClassifier(Classifier):
    def __init__(self):
        Classifier.__init__(self)
        self.cl = LogisticRegressionCV(Cs=1, penalty="l2", n_jobs=-1, random_state=0,
                                       class_weight={-1: 1, 1: 1},
                                       # class_weight="balanced",
                                       scoring=make_scorer(log_loss), solver="liblinear")

    def train(self, training_x, training_y):
        y = N.ravel(N.asarray(training_y))
        x = N.asarray(training_x)
        self.cl.fit(x, y)

    def test(self, testing_x):
        # Return the "probability" of the "correct" class (first label)
        # return N.clip(-self.cl.decision_function(testing_x), 0.00001, N.inf)
        return self.cl.predict_proba(testing_x)[:, 0]

class LinearRegressionClassifier(Classifier):
    def __init__(self):
        Classifier.__init__(self)
        self.cl = LinearRegression()

    def train(self, training_x, training_y):
        y = N.ravel(N.asarray(training_y))*10.0
        x = N.asarray(training_x)
        self.cl.fit(x, y)
        # print(self.cl.coef_)

    def test(self, testing_x):
        # Return the "probability" of the "correct" class (first label)
        # return self.cl.predict_proba(testing_x)[:, 0]
        return N.clip(self.cl.predict(testing_x)/10.0, 0.00001, 1.0)


class SGDRegressorClassifier(Classifier):
    def __init__(self):
        Classifier.__init__(self)
        self.cl = SGDClassifier(loss='log', penalty='l2',
                                learning_rate= 'optimal',
                                class_weight= 'balanced',
                                n_jobs=-1, random_state=0, max_iter= 1000)

    def train(self, training_x, training_y):
        y = N.ravel(N.asarray(training_y)) * 10.0
        x = N.asarray(training_x)
        self.cl.fit(x, y)
        # print(self.cl.coef_)

    def test(self, testing_x):
        # Return the "probability" of the "correct" class (first label)
        # return self.cl.predict_proba(testing_x)[:, 0]
        return N.clip(self.cl.predict(testing_x) / 10.0, 0.00001, 1.0)
