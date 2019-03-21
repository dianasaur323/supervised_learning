import models
import etl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics
import time
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    model_name = input("Model name?: ")
    data_set = input("Data set?: ")
    manual = input("Manual?: ")
    # weighted = input("Weighted?: ")
    df_test = None
    df_test_target = None
    df_test_new_target = None
    names = None
    if(data_set == 'reddit'):
        df_test = pd.read_csv('reddit_data/reddit_200k_test.csv',encoding='ISO-8859-1')
        df_test_new = pd.read_csv('reddit_data/reddit_200k_train.csv',encoding='ISO-8859-1')
        df_test_target = df_test['REMOVED'][:10000]
        df_test_new_target = df_test_new['REMOVED'][:10000]
        vectorizer = TfidfVectorizer()
        v = vectorizer.fit(df_test['body'])
        df_test = v.transform(df_test['body'])[:10000]
        df_test_new = v.transform(df_test_new['body'])[:10000]
        names = ['Not Removed', 'Removed']
        # if(weighted.toupper() == "Y"):
        #     for n in names:
        #         df_test['']
    else:
        data = fetch_20newsgroups(subset='train')
        df_test = data.data
        df_test_target = data.target
        # df_test_target = data.target[:1000]
        vectorizer = TfidfVectorizer()
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
        # print(df_test_new_target)
        # print(df_test_target)
        # print(names)
    if(model_name.upper() == "D"):
        model = models.DecisionTree()
        if manual.upper() == "Y":
            # print("HERE")
            params = {'max_depth': 100,
                      'min_samples_leaf': 200,
                      'class_weight':"balanced"}
                      # 'class_weight':"balanced"}
            start_time = time.time()
            model.clf.set_params(**params)
            model.clf.fit(df_test,df_test_target)
            df_test_new_pred = model.clf.predict(df_test_new)
            print("--- %s seconds ---" % (time.time() - start_time))

            models.plot_confusion_matrix(df_test_new_target, df_test_new_pred,names)
            models.plot_learning_curve(model.clf, "Decision Tree", \
                df_test, df_test_target, cv=5, train_sizes=np.arange(2000,5000,1000))
            print(metrics.classification_report(df_test_new_target, df_test_new_pred,target_names=names))
        else:
            # Plot validation curves for different parameters
            models.plot_validation_curve(model.clf,df_test,df_test_target,"min_samples_leaf",np.arange(1,200,50),5)
            models.plot_validation_curve(model.clf,df_test,df_test_target,"max_depth",np.arange(1,200,50),5)
            models.plot_validation_curve(model.clf,df_test,df_test_target,"min_impurity_decrease",np.linspace(.1,.3,5),5)

            # Grid search
            start_time = time.time()
            param_grid = {'max_depth': np.arange(1,200,50),
                          'min_samples_leaf': np.arange(1,200,50)}
                          # 'min_impurity_decrease':np.linspace(.1,.3,5)}
                          # 'class_weight':["balanced"]}
            gs_model = models.conduct_grid_search(model.clf, df_test, df_test_target, param_grid)
            print(gs_model.best_params_)
            print("--- %s seconds ---" % (time.time() - start_time))

            # Confusion Matrix
            df_test_new_pred = gs_model.best_estimator_.predict(df_test_new)
            # print(df_test_new_pred)
            models.plot_confusion_matrix(df_test_new_target, df_test_new_pred,names)

            # Learning curve
            models.plot_learning_curve(gs_model.best_estimator_, "Decision Tree", \
                df_test, df_test_target, cv=5, train_sizes=np.arange(2000,5000,1000))

            print(metrics.classification_report(df_test_new_target, df_test_new_pred,target_names=names))

    if(model_name.upper() == "N"):
        model = models.NeuralNetwork()
        # df_test = df_test
        # df_test_target = df_test_target
        # df_test_new = df_test_new
        # df_test_new_target = df_test_new
        if manual.upper() == "Y":
            params = {'hidden_layer_sizes': 3,
                      'batch_size': 1000,
                      'activation':'relu'}
                      # 'class_weight':"balanced"}
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

        else:
            df_test = df_test[:1000]
            df_test_target = df_test_target[:1000]
            df_test_new = df_test_new[:1000]
            df_test_new_target = df_test_new[:1000]
            # Plot validation curves for different parameters
            models.plot_validation_curve(model.clf,df_test,df_test_target,"hidden_layer_sizes",np.arange(1,4,1),5)
            models.plot_validation_curve(model.clf,df_test,df_test_target,"alpha",np.linspace(.01,.1,3),5)
            models.plot_validation_curve(model.clf,df_test,df_test_target,"batch_size",np.arange(200,1000,200),5)

            # param_grid = {'hidden_layer_sizes': np.arange(1,3,1),
            #               'batch_size':np.arange(200,1000,200),
            #               'activation':['relu','logistic','tanh'],
            #               'learning_rate':np.linspace(.001,.1,3)}

            # start_time = time.time()
            # gs_model = models.conduct_grid_search(model.clf, df_test, df_test_target, param_grid)
            # print(gs_model.best_params_)
            # print("--- %s seconds ---" % (time.time() - start_time))

            # Confusion Matrix
            # df_test_new_pred = gs_model.best_estimator_.predict(df_test_new)
            # models.plot_confusion_matrix(df_test_new_target, df_test_new_pred,names)
            #
            # models.plot_loss_curve(gd_model.best_estimator_)
            #
            # models.plot_learning_curve(gs_model.best_estimator_, "Neural Network", \
            #     df_test, df_test_target, cv=5, train_sizes=np.arange(2000,5000,3))
            #
            # print(metrics.classification_report(df_test_new_target, df_test_new_pred,target_names=names))

    if(model_name.upper() == "B"):
        model = models.Boosting()

        param_grid = {'n_estimators':np.arange(50,200,50),
                      'learning_rate':np.linspace(.001,.1,3)}

        # Plot validation curves for different parameters
        models.plot_validation_curve(model.clf,df_test,df_test_target,'learning_rate',np.linspace(.001,.1,3),3)
        models.plot_validation_curve(model.clf,df_test,df_test_target,'n_estimators',np.arange(50,200,50),3)

        params = {'n_estimators': 40,
                  'learning_rate': 0.05}
        start_time = time.time()
        model.clf.set_params(**params)
        model.clf.fit(df_test,df_test_target)
        df_test_new_pred = model.clf.predict(df_test_new)
        print("--- %s seconds ---" % (time.time() - start_time))

        # start_time = time.time()
        # gs_model = models.conduct_grid_search(model.clf, df_test, df_test_target, param_grid)
        # print(gs_model.best_params_)
        # print("--- %s seconds ---" % (time.time() - start_time))

        # Confusion Matrix
        models.plot_confusion_matrix(df_test_new_target, df_test_new_pred,names)


        models.plot_learning_curve(model.clf, "Boosting", \
            df_test, df_test_target, cv=5, train_sizes=np.arange(2000,5000,1000))

        print(metrics.classification_report(df_test_new_target, df_test_new_pred,target_names=names))

    if(model_name.upper() == "S"):
        model = models.SVM()
        df_test = StandardScaler(with_mean=False).fit_transform(df_test)
        df_test_new = StandardScaler(with_mean=False).fit_transform(df_test_new)

        models.plot_learning_curve(model.clf, "SVM", \
            df_test, df_test_target, cv=5, train_sizes=np.arange(2000,5000,1000))

        start_time = time.time()
        model.clf.fit(df_test,df_test_target)
        df_test_new_pred = model.clf.predict(df_test_new)
        print("--- %s seconds ---" % (time.time() - start_time))

        # Confusion Matrix
        models.plot_confusion_matrix(df_test_new_target, df_test_new_pred,names)

        # models.plot_loss_curve(model.clf)
        print(metrics.classification_report(df_test_new_target, df_test_new_pred,target_names=names))

    if(model_name.upper() == "K"):
        model = models.KNN()
        # df_test = df_test[:2000]
        # df_test_target = df_test_target[:2000]
        # df_test_new = df_test_new[:2000]
        # df_test_new_target = df_test_new[:2000]

        param_grid = {'n_neighbors': np.arange(0,40,5)}

        # Plot validation curves for different parameters
        models.plot_validation_curve(model.clf,df_test,df_test_target,'n_neighbors', np.arange(1,40,5),5)
        models.plot_validation_curve(model.clf,df_test,df_test_target,'weights',['uniform','distance'],5)

        # start_time = time.time()
        # gs_model = models.conduct_grid_search(model.clf, df_test, df_test_target, param_grid)
        # print(gs_model.best_params_)
        # print("--- %s seconds ---" % (time.time() - start_time))

        params = {'n_neighbors': 40,
                  'weights': 'distance'}
        start_time = time.time()
        model.clf.set_params(**params)
        model.clf.fit(df_test,df_test_target)
        df_test_new_pred = model.clf.predict(df_test_new)
        print("--- %s seconds ---" % (time.time() - start_time))

        models.plot_confusion_matrix(df_test_new_target, df_test_new_pred,names)
        models.plot_learning_curve(model.clf, "Decision Tree", \
            df_test, df_test_target, cv=5, train_sizes=np.arange(2000,5000,1000))
        print(metrics.classification_report(df_test_new_target, df_test_new_pred,target_names=names))

        # Confusion Matrix
        # df_test_new_pred = gs_model.best_estimator_.predict(df_test_new)
        # models.plot_confusion_matrix(df_test_new_target, df_test_new_pred,names)
        #
        # models.plot_learning_curve(gs_model.best_estimator_, "KNN", \
        #     df_test, df_test_target, cv=5, train_sizes=np.arange(2000,5000,1000))

        # print(metrics.classification_report(df_test_new_target, df_test_new_pred,target_names=names))
