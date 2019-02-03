import models
import pandas as pd

if __name__ == "__main__":
    model_name = input("Model name?: ")
    data_set = input("Data set?: ")
    if(data_set == 'reddit'):
        df_test = pd.read_csv('reddit_data/reddit_200k_test.csv',encoding='ISO-8859-1')
        df_train = pd.read_csv('reddit_data/reddit_200k_train.csv',encoding='ISO-8859-1')
    if(model_name.upper() == "DECISION TREE"):
        model = models.DecisionTree()
        print("decision tree")
