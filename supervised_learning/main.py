import models
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    model_name = input("Model name?: ")
    data_set = input("Data set?: ")
    if(data_set == 'reddit'):
        df_test = pd.read_csv('reddit_data/reddit_200k_test.csv',encoding='ISO-8859-1')
        df_test_class = df_test['REMOVED']
        df_test = df_test.drop(columns=['REMOVED','body','parent_id.x','id','created_utc.x','retrieved_on'])
        df_train = pd.read_csv('reddit_data/reddit_200k_train.csv',encoding='ISO-8859-1')
        print(df_train)
        df_train = df_train.drop(columns=['REMOVED','body','parent_id.x','id','created_utc.x','retrieved_on'])
        if(model_name.upper() == "DT"):
            model = models.DecisionTree()
            # model.run_model(df_test,df_test_class)
            # results = model.predict_model(df_train)
            # print(results)
            train_sizes, train_scores, valid_scores = model.plot_results(df_test,df_test_class)
            # print(train_sizes)
            # print(train_scores)
            # print(valid_scores)
            plt.plot(train_sizes, train_scores, train_sizes, valid_scores)
            plt.show()
        if(model_name.upper() == "N"):
            model = models.NeuralNetwork()
            train_sizes, train_scores, valid_scores = model.plot_results(df_test,df_test_class)
            plt.plot(train_sizes, train_scores, train_sizes, valid_scores)
            plt.show()
        if(model_name.upper() == "B"):
            model = models.Boosting()
            train_sizes, train_scores, valid_scores = model.plot_results(df_test,df_test_class)
            plt.plot(train_sizes, train_scores, train_sizes, valid_scores)
            plt.show()
        if(model_name.upper() == "S"):
            model = models.SVM()
            train_sizes, train_scores, valid_scores = model.plot_results(df_test,df_test_class)
            plt.plot(train_sizes, train_scores, train_sizes, valid_scores)
            plt.show()
        if(model_name.upper() == "K"):
            model = models.KNN()
            train_sizes, train_scores, valid_scores = model.plot_results(df_test,df_test_class)
            plt.plot(train_sizes, train_scores, train_sizes, valid_scores)
            plt.show()
