import models
import etl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics

if __name__ == "__main__":
    model_name = input("Model name?: ")
    data_set = input("Data set?: ")
    if(data_set == 'reddit'):
        df_test = pd.read_csv('reddit_data/reddit_200k_test.csv',encoding='ISO-8859-1')
        df_test_new = pd.read_csv('reddit_data/reddit_200k_train.csv',encoding='ISO-8859-1')
        df_test_target = df_test['REMOVED'][:10000]
        df_test_new_target = df_test_new['REMOVED'][:10000]
        vectorizer = TfidfVectorizer()
        v = vectorizer.fit(df_test['body'])
        df_test = v.transform(df_test['body'])[:10000]
        df_test_new = v.transform(df_test_new['body'])[:10000]
        if(model_name.upper() == "D"):
            model = models.DecisionTree()

            # Plot validation curves for different parameters
            models.plot_validation_curve(model.clf,df_test,df_test_target,"min_samples_leaf",np.arange(1,20,5),5)
            models.plot_validation_curve(model.clf,df_test,df_test_target,"max_depth",np.arange(1,20,5),5)
            models.plot_validation_curve(model.clf,df_test,df_test_target,"min_impurity_decrease",np.linspace(.1,.3,5),5)

            # Grid search
            start_time = time.time()
            param_grid = {'max_depth': np.arange(1,20,5),
                          'min_samples_leaf': np.arange(1,20,5),
                          'min_impurity_decrease':np.linspace(.1,.3,5)}
            gs_model = models.conduct_grid_search(model.clf, df_test, df_test_target, param_grid)
            print(gs_model.best_params_)
            print("--- %s seconds ---" % (time.time() - start_time))

            # Confusion Matrix
            df_test_new_pred = gs_model.best_estimator_.predict(df_test_new)
            models.plot_confusion_matrix(df_test_new_target, df_test_new_pred,["Not removed","Removed"])

            # Learning curve
            models.plot_learning_curve(gs_model.best_estimator_, "Decision Tree - Binary Classifier", \
                df_test, df_test_target, cv=5, train_sizes=np.arange(2000,5000,3))

            print(metrics.classification_report(df_test_new_target, df_test_new_pred,target_names=["Not removed","Removed"]))

        if(model_name.upper() == "N"):
            model = models.NeuralNetwork()

            # Plot validation curves for different parameters
            models.plot_validation_curve(model.clf,df_test,df_test_target,"hidden_layer_sizes",np.arange(1,3,1),5)
            models.plot_validation_curve(model.clf,df_test,df_test_target,"alpha",np.linspace(.01,.1,3),5)
            models.plot_validation_curve(model.clf,df_test,df_test_target,"batch_size",np.arange(200,1000,200),5)

            param_grid = {'hidden_layer_sizes': np.arange(1,3,1),
                          'alpha': np.linspace(.01,.1,3),
                          'batch_size':np.linspace(.1,.3,3),
                          'activation':['relu','logistic','tanh'],
                          'learning_rate':np.linspace(.001,.1,3)}

            start_time = time.time()
            gs_model = models.conduct_grid_search(model.clf, df_test, df_test_target, param_grid)
            print(gs_model.best_params_)
            print("--- %s seconds ---" % (time.time() - start_time))

            models.plot_loss_curve(gd_model.best_estimator_)

            models.plot_learning_curve(gs_model.best_estimator_, "Neural Network - Binary Classifier", \
                df_test, df_test_target, cv=5, train_sizes=np.arange(2000,5000,3))

            print(metrics.classification_report(df_test_new_target, df_test_new_pred,target_names=["Not removed","Removed"]))

        if(model_name.upper() == "B"):
            model = models.Boosting()

            param_grid = {'min_samples_leaf': np.arange(1,100,20),
                          'alpha': np.linspace(.01,.1,5),
                          'n_estimators':np.linspace(.1,.3,5),
                          'activation':['relu','logistic','tanh'],
                          'learning_rate':np.linspace(.001,.1,5)}

            # Plot validation curves for different parameters
            models.plot_validation_curve(model.clf,df_test,df_test_target,'min_samples_leaf', np.arange(20,100,20),5)
            models.plot_validation_curve(model.clf,df_test,df_test_target,'learning_rate',np.linspace(.001,.1,5),5)
            models.plot_validation_curve(model.clf,df_test,df_test_target,'n_estimators',np.linspace(.1,.3,5),5)

            start_time = time.time()
            gs_model = models.conduct_grid_search(model.clf, df_test, df_test_target, param_grid)
            print(gs_model.best_params_)
            print("--- %s seconds ---" % (time.time() - start_time))


            models.plot_learning_curve(gs_model.best_estimator_, "Boosting - Binary Classifier", \
                df_test, df_test_target, cv=5, train_sizes=np.arange(2000,5000,3))

            print(metrics.classification_report(df_test_new_target, df_test_new_pred,target_names=["Not removed","Removed"]))

        if(model_name.upper() == "S"):
            model = models.SVM()

            models.plot_learning_curve(model.clf, "SVM - Binary Classifier", \
                df_test, df_test_target, cv=5, train_sizes=np.arange(2000,5000,3))

            start_time = time.time()
            model.clf.fit(df_test,df_test_target)
            df_test_new_pred = model.clf.predict(df_test_new)
            print("--- %s seconds ---" % (time.time() - start_time))

            models.plot_loss_curve(model.clf)
            print(metrics.classification_report(df_test_new_target, df_test_new_pred,target_names=["Not removed","Removed"]))

        if(model_name.upper() == "K"):
            model = models.KNN()

            param_grid = {'n_neighbors': np.arange(1,20,5),
                          'p':[1,2]}

            # Plot validation curves for different parameters
            models.plot_validation_curve(model.clf,df_test,df_test_target,'n_neighbors', np.arange(1,20,5),5)
            models.plot_validation_curve(model.clf,df_test,df_test_target,'p',[1,2],5)

            start_time = time.time()
            gs_model = models.conduct_grid_search(model.clf, df_test, df_test_target, param_grid)
            print(gs_model.best_params_)
            print("--- %s seconds ---" % (time.time() - start_time))

            models.plot_learning_curve(gs_model.best_estimator_, "KNN - Binary Classifier", \
                df_test, df_test_target, cv=5, train_sizes=np.arange(2000,5000,3))

            print(metrics.classification_report(df_test_new_target, df_test_new_pred,target_names=["Not removed","Removed"]))
    else:
        print(fetch_20newsgroups.data)
