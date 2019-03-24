import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Leverages code from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# https://scikit-learn.org/0.15/auto_examples/model_selection/plot_confusion_matrix.html#example-model-selection-plot-confusion-matrix-py

# plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
# plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')
# plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')

def print_metrics(target, predicted, names):
    metrics.classification_report(target, predicted, names)
    pass

def plot_accuracy(title, xlabel, ylabel, xvalue, yvalue):
    plt.figure()
    plt.suptitle(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.xlim([2, 64])
    # plt.ylim([0, 1.0])
    # plt.plot(components, [baseline] * len(accuracies), color = "r")
    plt.plot(xvalue, yvalue)
    plt.show()
    pass

# def

# def plot_confusion_matrix(title='Confusion matrix', cmap=plt.cm.Blues, cm=, names=none):
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(names))
#     plt.xticks(tick_marks, names, rotation=45)
#     plt.yticks(tick_marks, names)
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     return plt

def plot_2d(estimator, title, X, y, cv, train_sizes):
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.scatter(X_std[km.labels == 0, 0], X_std[km.labels == 0, 1],
                c='green', label='cluster 1')
    plt.scatter(X_std[km.labels == 1, 0], X_std[km.labels == 1, 1],
                c='blue', label='cluster 2')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300,
                c='r', label='centroid')
    plt.legend()
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('Eruption time in mins')
    plt.ylabel('Waiting time to next eruption')
    plt.title('Visualization of clustered data', fontweight='bold')
    ax.set_aspect('equal');

def plot_learning_curve(estimator, title, X, y, cv, train_sizes):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training Size")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, train_sizes=train_sizes)
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
    return plt

# X_std = StandardScaler().fit_transform(df)
#
# # Run local implementation of kmeans
# km = Kmeans(n_clusters=2, max_iter=100)
# km.fit(X_std)
# centroids = km.centroids
#
# # Plot the clustered data
# fig, ax = plt.subplots(figsize=(6, 6))
# plt.scatter(X_std[km.labels == 0, 0], X_std[km.labels == 0, 1],
#             c='green', label='cluster 1')
# plt.scatter(X_std[km.labels == 1, 0], X_std[km.labels == 1, 1],
#             c='blue', label='cluster 2')
# plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300,
#             c='r', label='centroid')
# plt.legend()
# plt.xlim([-2, 2])
# plt.ylim([-2, 2])
# plt.xlabel('Eruption time in mins')
# plt.ylabel('Waiting time to next eruption')
# plt.title('Visualization of clustered data', fontweight='bold')
# ax.set_aspect('equal');
#
# gmm = GMM(n_components=4).fit(X)
# labels = gmm.predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');

# >>> import pylab as pl
# >>> iris = load_iris()
# >>> pca = PCA(n_components=2).fit(iris.data)
# >>> pca_2d = pca.transform(iris.data)
# >>> pl.figure('Reference Plot')
# >>> pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=iris.target)
# >>> kmeans = KMeans(n_clusters=3, random_state=111)
# >>> kmeans.fit(iris.data)
# >>> pl.figure('K-means with 3 clusters')
# >>> pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=kmeans.labels_)
# >>> pl.show()
