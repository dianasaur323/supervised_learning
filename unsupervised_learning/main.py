import models
import etl
import plot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics
import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from textblob import TextBlob
from spherecluster import SphericalKMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import SparseRandomProjection
from sklearn.random_projection import johnson_lindenstrauss_min_dim

def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words

def run_model(model_name, n_clusters, df_test, df_test_target):
    if(model_name.upper() == "K"):
        start_time = time.time()
        kmeans = KMeans(n_clusters=n_clusters,algorithm='full')
        kmeans.fit(df_test)
        print(kmeans.cluster_centers_)
        print(kmeans.labels_)
        print("--- %s seconds ---" % (time.time() - start_time))
        df_test_results = kmeans.predict(df_test)
        models.plot_confusion_matrix(df_test_target, df_test_results,names)
        print(metrics.classification_report(df_test_target, df_test_results,target_names=names))
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(df_test_target, kmeans.labels_))
        print("Completeness: %0.3f" % metrics.completeness_score(df_test_target, kmeans.labels_))
        print("V-measure: %0.3f" % metrics.v_measure_score(df_test_target, kmeans.labels_))
        print("Adjusted Rand-Index: %.3f"
              % metrics.adjusted_rand_score(df_test_target, kmeans.labels_))
        print("Silhouette Coefficient (E): %0.3f"
              % metrics.silhouette_score(df_test, kmeans.labels_, sample_size=1000))
        print("Silhouette Coefficient (C): %0.3f"
              % metrics.silhouette_score(df_test, kmeans.labels_, metric = 'cosine', sample_size=1000))
        print("Inertia: %0.3f"
            % kmeans.inertia_)
        print("Top terms per cluster:")
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        for i in range(2):
            top_ten_words = [terms[ind] for ind in order_centroids[i, :5]]
            print("Cluster {}: {}".format(i, ' '.join(top_ten_words)))
        return kmeans
    elif(model_name.upper() == "S"):
        start_time = time.time()
        skm = SphericalKMeans(n_clusters=n_clusters)
        skm.fit(df_test)
        df_test_results = skm.predict(df_test)
        models.plot_confusion_matrix(df_test_target, df_test_results,names)
        print(metrics.classification_report(df_test_target, df_test_results,target_names=names))
        print(skm.cluster_centers_)
        print(skm.labels_)
        print(skm.inertia_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(df_test_target, skm.labels_))
        print("Completeness: %0.3f" % metrics.completeness_score(df_test_target, skm.labels_))
        print("V-measure: %0.3f" % metrics.v_measure_score(df_test_target, skm.labels_))
        print("Adjusted Rand-Index: %.3f"
              % metrics.adjusted_rand_score(df_test_target, skm.labels_))
        print("Silhouette Coefficient (E): %0.3f"
              % metrics.silhouette_score(df_test, skm.labels_, sample_size=1000))
        print("Silhouette Coefficient (C): %0.3f"
              % metrics.silhouette_score(df_test, skm.labels_, metric = 'cosine', sample_size=1000))
        print("Top terms per cluster:")
        order_centroids = skm.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        for i in range(2):
            top_ten_words = [terms[ind] for ind in order_centroids[i, :5]]
            print("Cluster {}: {}".format(i, ' '.join(top_ten_words)))
        return skm
    elif(model_name.upper() == "EM"):
        start_time = time.time()
        gmm = GaussianMixture(n_components=n_clusters)
        gmm.fit(df_test.toarray())
        df_test_results = gmm.predict(df_test)
        models.plot_confusion_matrix(df_test_target, df_test_results,names)
        print(metrics.classification_report(df_test_target, df_test_results,target_names=names))
        print("Means: %0.3f" % gmm.means_)
        print("Means: %0.3f" % gmm.covariances_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(df_test_target, skm.labels_))
        print("Completeness: %0.3f" % metrics.completeness_score(df_test_target, skm.labels_))
        print("V-measure: %0.3f" % metrics.v_measure_score(df_test_target, skm.labels_))
        print("Adjusted Rand-Index: %.3f"
              % metrics.adjusted_rand_score(df_test_target, skm.labels_))
        print("Silhouette Coefficient (E): %0.3f"
              % metrics.silhouette_score(df_test, skm.labels_, sample_size=1000))
        print("Silhouette Coefficient (C): %0.3f"
              % metrics.silhouette_score(df_test, skm.labels_, metric = 'cosine', sample_size=1000))
        print("Top terms per cluster:")
        order_centroids = skm.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        for i in range(2):
            top_ten_words = [terms[ind] for ind in order_centroids[i, :5]]
            print("Cluster {}: {}".format(i, ' '.join(top_ten_words)))
        return gmm


if __name__ == "__main__":
    model_name = input("Model name?: ")
    data_set = input("Data set?: ")
    feature_selection = input("Feature selection?: ")
    df_test = None
    df_test_target = None
    df_test_new_target = None
    names = None
    n_clusters = 0
    if(data_set == 'reddit'):
        df_test = pd.read_csv('reddit_data/reddit_200k_test.csv',encoding='ISO-8859-1')
        df_test_new = pd.read_csv('reddit_data/reddit_200k_train.csv',encoding='ISO-8859-1')
        df_test_target = df_test['REMOVED'][:10000]
        # df_test_new_target = df_test_new['REMOVED'][:10000]
        # vectorizer = TfidfVectorizer(tokenizer=textblob_tokenizer, stop_words = 'english')
        # vectorizer = TfidfVectorizer(stop_words = 'english', max_features=40)
        vectorizer = TfidfVectorizer(stop_words = 'english')
        v = vectorizer.fit(df_test['body'])
        df_test = v.transform(df_test['body'])[:10000]
        df_test_new = v.transform(df_test_new['body'])[:10000]
        names = ['Not Removed', 'Removed']
        n_clusters=2
    else:
        data = fetch_20newsgroups(subset='train')
        df_test = data.data
        df_test_target = data.target
        # df_test_target = data.target[:1000]
        vectorizer = TfidfVectorizer(stop_words = 'english')
        v = vectorizer.fit(df_test)
        df_test = v.transform(df_test)
        # df_test = v.transform(df_test)[:1000]
        # data = fetch_20newsgroups(subset='test',categories=['comp.graphics'])
        data = fetch_20newsgroups(subset='test')
        df_test_new = data.data
        df_test_new = v.transform(df_test_new)
        # df_test_new = v.transform(df_test_new)[:1000]
        df_test_new_target = data.target
        # df_test_new_target = data.target[:1000]
        names=data.target_names
        n_clusters=20
        # print(df_test_new_target)
        # print(df_test_target)
        # print(names)
    if(feature_selection.upper() == "PCA"):
        start_time = time.time()
        pca = PCA(n_components=n_clusters)
        df_test = pca.fit_transform(df_test.toarray())
        print("--- Feature Selection: %s seconds ---" % (time.time() - start_time))
    elif(feature_selection.upper() == "ICA"):
        start_time = time.time()
        ica = FastICA(n_components=n_clusters)
        df_test = ica.fit_transform(df_test)
        print("--- Feature Selection: %s seconds ---" % (time.time() - start_time))
    elif(feature_selection.upper() = "RP"):
        start_time = time.time()
        rp = SparseRandomProjection()
        df_test = rp.fit_transform(df_test)
        print("--- Feature Selection: %s seconds ---" % (time.time() - start_time))
        print(johnson_lindenstrauss_min_dim(1797,eps=0.1))
        if(feature_selection.upper() = "RP"):
            accuracies = []
            components = np.int32(np.linspace(2, 64, 20))
            for comp in components:
                 rp = SparseRandomProjection(n_components = comp)
                 df_test = rp.fit_transform(df_test)
                 model = run_model(model_name, n_clusters, df_test, df_test_target)
                 accuracies.append(metrics.accuracy_score(model.predict(df_test), df_test_target))
            plot.plot_accuracy(title="RP Accuracies", "n_components", "accuracy_score", components, accuracies)
    else:
        run_model(model_name, n_clusters, df_test, df_test_target)
