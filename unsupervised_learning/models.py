from sklearn import tree
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve
import numpy as np
import plot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time

# MULTI CLASS RESEARCH: https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
# http://people.cs.pitt.edu/~milos/courses/cs2750-Spring2014/Lectures/Class13.pdf
# https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5
# https://scikit-learn.org/stable/auto_examples/plot_multilabel.html
# https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/

# CHARTING
# https://scikit-learn.org/stable/modules/learning_curve.html
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

# https://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html

def conduct_grid_search(estimator,train,target,param_grid):
    # param_grid = {'max_depth': np.arange(3, 10)}
    clf = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5)
    clf.fit(train, target)
    return(clf)

def plot_validation_curve(estimator,train,target,param_name,param_range,cv):
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py
    train_scores,test_scores = validation_curve(estimator=estimator,X=train, \
        y=target,param_name=param_name,param_range=param_range,cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title("Validation Curve")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    lw = 2
    # plt.semilogx(param_range, train_scores_mean, label="Training score",
    #              color="darkorange", lw=lw)
    plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    # plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
    #              color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()

def plot_learning_curve(estimator, title, train, target, cv, train_sizes):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training Size")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, train, target, cv=cv, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()

# https://scikit-learn.org/0.15/auto_examples/model_selection/plot_confusion_matrix.html#example-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(test, test_target, names):
    # print(test)
    # print(test_target)
    cm = confusion_matrix(test, test_target)
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # cm = cm.astype('float') / cm.sum(axis=1)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_loss_curve(estimator):
    # plt.plot(scores_train, color='green', alpha=0.8, label='Train')
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Loss Curve")
    plt.plot(estimator.loss_curve_)
    plt.show()

class DecisionTree:
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
    # RESEARCH: https://www.bogotobogo.com/python/scikit-learn/scikit_machine_learning_Constructing_Decision_Tree_Learning_Information_Gain_IG_Impurity_Entropy_Gini_Classification_Error.php
    # https://www.kdnuggets.com/2017/05/simplifying-decision-tree-interpretation-decision-rules-python.html
    # https://www.bogotobogo.com/python/scikit-learn/scikit_machine_learning_Constructing_Decision_Tree_Learning_Information_Gain_IG_Impurity_Entropy_Gini_Classification_Error.php
    # https://scikit-learn.org/stable/modules/tree.html
    # https://www.datascience.com/blog/random-forests-decision-trees-ensemble-methods
    # http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/
    # https://stackoverflow.com/questions/48090757/text-classification-using-decision-trees-in-python
    # https://stackoverflow.com/questions/49428469/pruning-decision-trees
    def __init__(self):
        self.clf = tree.DecisionTreeClassifier()

    def run_model(self,x,y):
        self.clf.fit(x,y)

    def predict_model(self,x):
        return self.clf.predict(x)

    def plot_results(self,x,y):
        title = "Decision Tree - Reddit Data Set"
        return plot.plot_learning_curve(self.clf, title, x, y, cv=5, train_sizes=np.linspace(.1, 1.0))

class NeuralNetwork:
    # https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    def __init__(self):
        self.clf = MLPClassifier()

class Boosting:
    # Research: https://towardsdatascience.com/decision-tree-ensembles-bagging-and-boosting-266a8ba60fd9
    # Example of Gradient Boosting https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regularization.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regularization-py
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#examples-using-sklearn-ensemble-gradientboostingclassifier
    # def __init__(self):
    #     self.params = {'n_estimators': 1000, 'max_leaf_nodes': 4, 'max_depth': None, 'random_state': 2,
    #                'min_samples_split': 5}
    #     self.clf = GradientBoostingClassifier(**self.params)

    def __init__(self):
        self.params_dt = {'max_depth': 100,
                  'min_samples_leaf': 200,
                  'class_weight':"balanced"}
        self.dt = tree.DecisionTreeClassifier(**self.params_dt)
        self.params = {'base_estimator':self.dt}
        self.clf = AdaBoostClassifier(**self.params)

class SVM:
    def __init__(self):
        self.clf = svm.SVC(gamma='scale',max_iter=200,class_weight='balanced')

class KNN:
    # https://scikit-learn.org/stable/modules/neighbors.html
    # EXAMPLE: https://shankarmsy.github.io/stories/knn-sklearn.html
    # https://shankarmsy.github.io/stories/knn-sklearn.html
    def __init__(self):
        self.clf = KNeighborsClassifier()
