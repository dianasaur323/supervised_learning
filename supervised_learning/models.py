from sklearn import tree
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.neighbors import NearestNeighbors


class DecisionTree:
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
    def __init__(self):
        self.clf = tree.DecisionTreeClassifier()

    def run_model(self,x,y):
        self.clf.fit(x,y)

    def predict_model(self,x):
        return self.clf.predict(x)

    def plot_results(self,x,y):
        return learning_curve(self.clf, x, y, train_sizes=[1000, 2000, 3000], cv=5)

class NeuralNetwork:
    # https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    def __init__(self):
        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

    def plot_results(self,x,y):
        return learning_curve(self.clf, x, y, train_sizes=[1000, 2000, 3000,5000,30000], cv=5)

class Boosting:
    # Research: https://towardsdatascience.com/decision-tree-ensembles-bagging-and-boosting-266a8ba60fd9
    # Example of Gradient Boosting https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regularization.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regularization-py
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#examples-using-sklearn-ensemble-gradientboostingclassifier
    def __init__(self):
        self.params = {'n_estimators': 1000, 'max_leaf_nodes': 4, 'max_depth': None, 'random_state': 2,
                   'min_samples_split': 5}
        self.clf = GradientBoostingClassifier(**self.params)

    def plot_results(self,x,y):
        return learning_curve(self.clf, x, y, train_sizes=[1000, 2000, 3000,5000,30000], cv=5)

class SVM:
    def __init__(self):
        self.clf = svm.SVC(gamma='scale')

    def plot_results(self,x,y):
        return learning_curve(self.clf, x, y, train_sizes=[1000, 2000, 3000,5000,30000], cv=5)

class KNN:
    # https://scikit-learn.org/stable/modules/neighbors.html
    # EXAMPLE: https://shankarmsy.github.io/stories/knn-sklearn.html
    def __init__(self):
        self.clf = NearestNeighbors(n_neighbors=2)

    def plot_results(self,x,y):
        return learning_curve(self.clf, x, y, train_sizes=[1000, 2000, 3000,5000,30000], cv=5)
