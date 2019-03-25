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
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import kurtosis
import statistics
from sklearn.decomposition import LatentDirichletAllocation

def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words

def run_model(model_name, n_clusters, df_test, df_test_target):
    # Nc = range(1, 20)
    # kmeans = [KMeans(n_clusters=i) for i in Nc]
    # kmeans
    # score = [kmeans[i].fit(Y_norm).score(Y_norm) for i in range(len(kmeans))]
    # score
    # pl.plot(Nc,score)
    # pl.xlabel('Number of Clusters')
    # pl.ylabel('Score')
    # pl.title('Elbow Curve')
    # pl.show()
    if(model_name.upper() == "K"):
        start_time = time.time()
        # kmeans = KMeans(init='k-means++',n_clusters=n_clusters,algorithm='full')
        kmeans = KMeans(init='random',n_clusters=n_clusters,algorithm='full')
        # kmeans = KMeans(init='random',n_clusters=n_clusters,algorithm='full',n_init=30)
        kmeans.fit(df_test)
        # print(kmeans.cluster_centers_)
        # print(kmeans.labels_)
        print("--- k-means: %s seconds ---" % (time.time() - start_time))
        df_test_results = kmeans.predict(df_test)
        models.plot_confusion_matrix(df_test_target, df_test_results,names)
        print(metrics.classification_report(df_test_target, df_test_results,target_names=names))
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(df_test_target, kmeans.labels_))
        print("Completeness: %0.3f" % metrics.completeness_score(df_test_target, kmeans.labels_))
        print("V-measure: %0.3f" % metrics.v_measure_score(df_test_target, kmeans.labels_))
        print("Adjusted Rand-Index: %.3f"
              % metrics.adjusted_rand_score(df_test_target, kmeans.labels_))
        print("AMI: %.3f"
              % metrics.adjusted_mutual_info_score(df_test_target, kmeans.labels_))
        print("Silhouette Coefficient (E): %0.3f"
              % metrics.silhouette_score(df_test, kmeans.labels_, sample_size=1000))
        print("Silhouette Coefficient (C): %0.3f"
              % metrics.silhouette_score(df_test, kmeans.labels_, metric = 'cosine', sample_size=1000))
        print("Accuracy Score: %0.3f"
              % metrics.accuracy_score(df_test_results, df_test_target))
        print("Top terms per cluster:")
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        if(feature_selection.upper() == ''):
            Nc = range(len(names), len(names)*5,10)
            kmeans = [KMeans(init='random',n_clusters=i) for i in Nc]
            score = [metrics.silhouette_score(df_test, kmeans[i].fit(df_test).labels_, sample_size=1000) for i in range(len(kmeans))]
            plt.plot(Nc,score)
            plt.xlabel('Number of Clusters')
            plt.ylabel('Score')
            plt.title('Elbow Curve - Multi-Class')
            plt.show()
        terms = vectorizer.get_feature_names()
        for i in range(len(names)):
            top_ten_words = [terms[ind] for ind in order_centroids[i, :5]]
            print("Cluster {}: {}".format(i, ' '.join(top_ten_words)))
        return kmeans
    elif(model_name.upper() == "S"):
        start_time = time.time()
        skm = SphericalKMeans(n_clusters=n_clusters)
        skm.fit(df_test)
        df_test_results = skm.predict(df_test)
        print("--- skm: %s seconds ---" % (time.time() - start_time))
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
        print("AMI: %.3f"
              % metrics.adjusted_mutual_info_score(df_test_target, skm.labels_))
        print("Silhouette Coefficient (E): %0.3f"
              % metrics.silhouette_score(df_test, skm.labels_, sample_size=1000))
        print("Silhouette Coefficient (C): %0.3f"
              % metrics.silhouette_score(df_test, skm.labels_, metric = 'cosine', sample_size=1000))
        print("Top terms per cluster:")
        order_centroids = skm.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        for i in range(len(names)):
            top_ten_words = [terms[ind] for ind in order_centroids[i, :5]]
            print("Cluster {}: {}".format(i, ' '.join(top_ten_words)))
        return skm
    elif(model_name.upper() == "EM"):
        start_time = time.time()
        # gmm = GaussianMixture(n_components=n_clusters)
        # gmm = GaussianMixture(n_components=n_clusters, init_params = 'kmeans')
        # gmm = GaussianMixture(n_components=n_clusters, init_params = 'random')
        # gmm = GaussianMixture(n_components=n_clusters, init_params = 'kmeans', n_init=5)
        # gmm = GaussianMixture(n_components=n_clusters, init_params = 'random', n_init=5)
        # gmm = GaussianMixture(n_components=n_clusters, init_params = 'kmeans', n_init=5, covariance_type='full')
        # gmm = GaussianMixture(n_components=n_clusters, init_params = 'kmeans', n_init=5, covariance_type='tied')
        # gmm = GaussianMixture(n_components=n_clusters, init_params = 'kmeans', n_init=5, covariance_type='diag')
        gmm = GaussianMixture(n_components=n_clusters, init_params = 'kmeans', n_init=5, covariance_type='spherical')
        gmm.fit(df_test.toarray())
        df_test_results = gmm.predict(df_test.toarray())
        print("--- gmm: %s seconds ---" % (time.time() - start_time))
        models.plot_confusion_matrix(df_test_target, df_test_results,names)
        print(metrics.classification_report(df_test_target, df_test_results,target_names=names))
        print("AIC: %0.3f" % gmm.aic(df_test.toarray()))
        print("BIC: %0.3f" % gmm.bic(df_test.toarray()))

        # models = [GMM(n, covariance_type='full', random_state=0).fit(Xmoon)
        #   for n in n_components]
        #     plt.plot(n_components, [m.bic(Xmoon) for m in models], label='BIC')
        #     plt.plot(n_components, [m.aic(Xmoon) for m in models], label='AIC')
        #     plt.legend(loc='best')
        #     plt.xlabel('n_components');
        # print(gmm.covariances_)
        # print(gmm.precisions_)
        print("Accuracy: %0.3f" % metrics.accuracy_score(gmm.predict(df_test.toarray()), df_test_target))
        print(metrics.classification_report(df_test_target, df_test_results,target_names=names))
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(df_test_target, df_test_results))
        print("Completeness: %0.3f" % metrics.completeness_score(df_test_target, df_test_results))
        print("V-measure: %0.3f" % metrics.v_measure_score(df_test_target, df_test_results))
        print("Adjusted Rand-Index: %.3f"
              % metrics.adjusted_rand_score(df_test_target, df_test_results))
        print("AMI: %.3f"
              % metrics.adjusted_mutual_info_score(df_test_target, df_test_results))
        print("Silhouette Coefficient (E): %0.3f"
              % metrics.silhouette_score(df_test, df_test_results, sample_size=1000))
        print("Silhouette Coefficient (C): %0.3f"
              % metrics.silhouette_score(df_test, df_test_results, metric = 'cosine', sample_size=1000))
        if(feature_selection.upper() == ''):
            n_dimensions = range(len(names), len(names)*10,10)
            gmm = [GaussianMixture(n_components=i) for i in n_dimensions]
            AIC = [gmm[i].fit(df_test.toarray()).aic(df_test.toarray()) for i in range(len(gmm))]
            BIC = [gmm[i].fit(df_test.toarray()).bic(df_test.toarray()) for i in range(len(gmm))]
            plt.plot(n_dimensions,AIC)
            plt.plot(n_dimensions,BIC)
            plt.xlabel('Number of Components')
            plt.ylabel('AIC / BIC')
            plt.title('Multi-Class')
            plt.show()
        return gmm
    elif(model_name.upper() == "N"):
        model = models.NeuralNetwork()
        params = {'hidden_layer_sizes': 3,
                  'batch_size': 1000,
                  'activation':'relu'}
        model.clf.set_params(**params)
        start_time = time.time()
        model.clf.set_params(**params)
        model.clf.fit(df_test,df_test_target)
        df_test_new_pred = model.clf.predict(df_test_new)
        print("--- %s seconds ---" % (time.time() - start_time))

        models.plot_confusion_matrix(df_test_new_target, df_test_new_pred,names)
        models.plot_learning_curve(model.clf, "Decision Tree", \
            df_test, df_test_target, cv=5, train_sizes=np.arange(2000,5000,1000))
        print(metrics.classification_report(df_test_new_target, df_test_new_pred,target_names=names))

        models.plot_loss_curve(model.clf)

        models.plot_learning_curve(model.clf, "Neural Network", \
            df_test, df_test_target, cv=5, train_sizes=np.arange(2000,5000,1000))

        print(metrics.classification_report(df_test_new_target, df_test_new_pred,target_names=names))
        return model


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
        if(model_name.upper == "EM"): vectorizer = TfidfVectorizer(stop_words = 'english', max_features=200)
        # vectorizer = TfidfVectorizer()
        elif(feature_selection != '' and feature_selection.upper() != 'LDA'): vectorizer = TfidfVectorizer(stop_words = 'english', max_features=200)
        else: vectorizer = TfidfVectorizer(stop_words = 'english')
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
        if(model_name.upper == "EM"): vectorizer = TfidfVectorizer(stop_words = 'english', max_features=200)
        elif(feature_selection != '' and feature_selection.upper() != 'LDA'): vectorizer = TfidfVectorizer(stop_words = 'english', max_features=200)
        else: vectorizer = TfidfVectorizer(stop_words = 'english')
        # vectorizer = TfidfVectorizer(stop_words = 'english', max_features=200)
        # vectorizer = TfidfVectorizer()
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
        if(model_name != '' and data_set != ''):
            pca = PCA(n_components=58)
            df_test = pca.fit_transform(df_test.toarray())
            run_model(model_name, n_clusters, df_test, df_test_target)
        elif(model_name != ''):
            pca = PCA(n_components=53)
            df_test = pca.fit_transform(df_test.toarray())
            run_model(model_name, n_clusters, df_test, df_test_target)
        else:
            start_time = time.time()
            pca = PCA(n_components=2)
            df_test_pca = pca.fit_transform(df_test.toarray())
            print("--- Feature Selection: %s seconds ---" % (time.time() - start_time))
            print(pca.explained_variance_)
            plt.scatter(df_test_pca[:, 0], df_test_pca[:, 1],
                c=df_test_target, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('rainbow', 10))
            # print(df_test_target)
            plt.suptitle("PCA 2D: Multi-Class")
            plt.xlabel('component 1')
            plt.ylabel('component 2')
            plt.colorbar();
            plt.show();

            if(data_set == ''):
                pca = PCA(n_components=3)
                df_test_pca = pca.fit_transform(df_test.toarray())
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.scatter(df_test_pca[:, 0], df_test_pca[:, 1], df_test_pca[:, 2],
                    c = df_test_target, edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('rainbow', 10))
                plt.suptitle("PCA 3D: Multi-Class")
                plt.show();

            pca = PCA().fit(df_test.toarray())
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('n_components')
            plt.ylabel('cumulative explained variance');
            # plt.suptitle("PCA Variance: Binary")
            plt.suptitle("PCA Variance: Multi-Class")
            plt.show()

            # pca = PCA(0.50).fit(df_test.toarray())
            # print(pca.n_components_)

            if(data_set == 'reddit'):
                pca = PCA(n_components=53).fit(df_test.toarray())
                df_test_centered = df_test.toarray() - np.mean(df_test.toarray(), axis=0)
                cov_matrix = np.dot(df_test_centered.T, df_test_centered) / df_test.shape[0]
                eigenvalues = pca.explained_variance_
                print(eigenvalues)
                plt.hist(eigenvalues, color = 'blue')
                plt.title('Eigenvalue Distribution - Binary')
                plt.show()
                df_test_pca = pca.fit_transform(df_test.toarray())
                df_test_pca_inverse = pca.inverse_transform(df_test_pca)
                new_pca = PCA(n_components=2)
                df_test_pca = new_pca.fit_transform(df_test_pca_inverse)
                plt.scatter(df_test_pca[:, 0], df_test_pca[:, 1],
                    c=df_test_target, edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('rainbow', 10))
                # print(df_test_target)
                plt.suptitle("PCA 2D: Binary")
                plt.xlabel('component 1')
                plt.ylabel('component 2')
                plt.colorbar();
                plt.show();
            else:
                pca = PCA(n_components=58).fit(df_test.toarray())
                df_test_centered = df_test.toarray() - np.mean(df_test.toarray(), axis=0)
                cov_matrix = np.dot(df_test_centered.T, df_test_centered) / df_test.shape[0]
                eigenvalues = pca.explained_variance_
                print(eigenvalues)
                plt.hist(eigenvalues, color = 'blue')
                plt.title('Eigenvalue Distribution - Multi-Class')
                plt.show()
                df_test_pca = pca.fit_transform(df_test.toarray())
                df_test_pca_inverse = pca.inverse_transform(df_test_pca)
                new_pca = PCA(n_components=2)
                df_test_pca = new_pca.fit_transform(df_test_pca_inverse)
                plt.scatter(df_test_pca[:, 0], df_test_pca[:, 1],
                    c=df_test_target, edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('rainbow', 10))
                # print(df_test_target)
                plt.suptitle("PCA 2D: Multi-Class")
                plt.xlabel('component 1')
                plt.ylabel('component 2')
                plt.colorbar();
                plt.show();
        # for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):
        #     print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
        #     print(eigenvalue)
        # variance = []
        # components = np.int32(np.linspace(2, 40 , 2))
        # for comp in components:
        #      pca = PCA(n_components=comp)
        #      df_test_pca = pca.fit_transform(df_test.toarray())
        #      variance.append(max(pca.explained_variance_))
        # plot.plot_accuracy(title="PCA Max Variance - Multi-Class",xlabel="n_dimensions", ylabel="variance", xvalue=components, yvalue=variance)
    elif(feature_selection.upper() == "ICA"):
        start_time = time.time()
        ica = FastICA(n_components=len(names))
        df_test_ica = ica.fit_transform(df_test.toarray())
        print("--- Feature Selection: %s seconds ---" % (time.time() - start_time))
        if(model_name != ''):
            ica = FastICA(n_components=len(names))
            df_test = ica.fit_transform(df_test.toarray())
            run_model(model_name, n_clusters, df_test, df_test_target)
        else:
            if(data_set == 'reddit'):
                plt.scatter(df_test_ica[:, 0], df_test_ica[:, 1],
                    c=df_test_target, edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('rainbow', 10))
                # plt.suptitle("ICA 2D: Binary")
                plt.plot(df_test_ica[0])
                plt.plot(df_test_ica[1])
                plt.suptitle("ICA 2D: Binary")
                plt.xlabel('component 1')
                plt.ylabel('component 2')
                plt.colorbar();
                plt.show();
                print(kurtosis(df_test_ica))
                plt.hist(df_test_ica)
                plt.title('ICA distribution - Binary')
                plt.show()
                kurtosis_list = []
                components = np.int32(np.linspace(2, 100 , 10))
                for comp in components:
                    ica = FastICA(n_components=comp)
                    df_test_ica = ica.fit_transform(df_test.toarray())
                    kurtosis_list.append(max(kurtosis(df_test_ica)))
                plot.plot_accuracy(title="ICA Max Kurtosis - Binary",xlabel="n_components", ylabel="kurtosis", xvalue=components, yvalue=kurtosis_list)
            else:
                print(kurtosis(df_test_ica))
                plt.hist(df_test_ica)
                plt.title('ICA distribution - Multi-class')
                plt.show()
                kurtosis_list = []
                components = np.int32(np.linspace(20, 200 , 20))
                for comp in components:
                    ica = FastICA(n_components=comp)
                    df_test_ica = ica.fit_transform(df_test.toarray())
                    kurtosis_list.append(max(kurtosis(df_test_ica)))
                plot.plot_accuracy(title="ICA Max Kurtosis - Multi-class",xlabel="n_components", ylabel="kurtosis", xvalue=components, yvalue=kurtosis_list)
    elif(feature_selection.upper() =="RP"):
        start_time = time.time()
        rp = SparseRandomProjection(len(names))
        df_test = rp.fit_transform(df_test)
        print("--- Feature Selection: %s seconds ---" % (time.time() - start_time))
        if(model_name != ''):
            accuracies = []
            components = np.int32(np.linspace(2, 64, 20))
            for comp in components:
                 rp = SparseRandomProjection(n_components = comp)
                 df_test = rp.fit_transform(df_test)
                 model = run_model(model_name, n_clusters, df_test, df_test_target)
                 accuracies.append(metrics.v_measure_score(df_test_target, model.predict(df_test)))
            plot.plot_accuracy(title="RP Accuracies",xlabel="n_components", ylabel="V-Measure", xvalue=components, yvalue=accuracies)
        else:
            print(rp.n_components_)
            print(df_test.mean())
    elif(feature_selection.upper() == "LDA"):
        lda = LatentDirichletAllocation(n_components = len(names))
        df_test = lda.fit_transform(df_test)
        if(model_name != ''):
            run_model(model_name, n_clusters, df_test, df_test_target)
    else:
        run_model(model_name, n_clusters, df_test, df_test_target)
