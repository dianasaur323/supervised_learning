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
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import mlrose
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import math

if __name__ == "__main__":
    model_name = input("Model name?: ")
    data_set = input("Data set?: ")
    run = input("Run Neural?: ")
    # weighted = input("Weighted?: ")
    df_test = None
    df_test_target = None
    df_test_new_target = None
    names = None
    if(data_set == 'reddit'):
        df_test = pd.read_csv('reddit_data/reddit_200k_test.csv',encoding='ISO-8859-1')[:3000]
        df_train = pd.read_csv('reddit_data/reddit_200k_train.csv',encoding='ISO-8859-1')[:3000]
        # df_test_target = df_test['REMOVED']
        df_class_0 = resample(df_test[df_test['REMOVED'] == False],random_state=0,n_samples=int(len(df_test)/2), replace=True)
        df_class_1 = resample(df_test[df_test['REMOVED'] == True],random_state=0,n_samples=int(len(df_test)/2))
        df_test = df_class_0.append(df_class_1)
        # print(df_test)
        df_test_target = df_test['REMOVED']
        # print(df_test_target)
        df_train_target = df_train['REMOVED']
        vectorizer = TfidfVectorizer()
        v = vectorizer.fit(df_test['body'])
        df_test = v.transform(df_test['body'])
        df_train = v.transform(df_train['body'])
        names = ['Not Removed', 'Removed']
        # if(weighted.toupper() == "Y"):
        #     for n in names:
        #         df_test['']
    if(run.upper() == "Y"):
        # print(df_test.shape)
        # print(len(df_test_target))
        # print(df_test_target)
        # one_hot = OneHotEncoder()
        # df_test_target = one_hot.transform(df_test).toarray().todense()
        # y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()
        np.random.seed(3)
        df_test = df_test.toarray()
        # A[np.random.choice(A.shape[0], num_rows_2_sample, replace=False)]
        df_test_target = df_test_target.tolist()
        df_train = df_train.toarray()
        df_train_target = df_train_target.tolist()
        # scaler = MinMaxScaler()
        # df_train = scaler.fit_transform(df_train)
        # df_test = scaler.transform(df_test)
        start_time = time.time()
        nn_model1 = None
        if(model_name.upper() == "R"):
            nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [3], activation = 'relu',
                                     algorithm = 'random_hill_climb', max_iters = 1000,
                                     bias = True, is_classifier = True, learning_rate = 0.01,
                                     early_stopping = True, max_attempts = 100,
                                     clip_max = 5)
        elif(model_name.upper() == "S"):
            geom = mlrose.decay.GeomDecay(init_temp=1.0, decay=0.99999, min_temp=0.001)
            nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [3], activation = 'relu',
                                     # algorithm = 'simulated_annealing', schedule = geom, max_iters = 500,
                                     algorithm = 'simulated_annealing', max_iters = 1000,
                                     bias = True, is_classifier = True, learning_rate = 0.01,
                                     early_stopping = True, max_attempts = 100)
        elif(model_name.upper() == "G"):
            nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [3], activation = 'relu',
                                     algorithm = 'genetic_alg', max_iters = 1000, pop_size = 200,
                                     bias = True, is_classifier = True, learning_rate = 0.01,
                                     early_stopping = True, max_attempts = 100,
                                     clip_max = 5)
        else:
            nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [3], activation = 'relu',
                                     algorithm = 'gradient_descent', max_iters = 1000,
                                     bias = True, is_classifier = True, learning_rate = 0.001,
                                     early_stopping = True, max_attempts = 100)
        nn_model1.fit(df_test, df_test_target)
        print("--- FIT: %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        df_test_pred = nn_model1.predict(df_test)
        print("--- PREDICT: %s seconds ---" % (time.time() - start_time))
        df_test_acc = accuracy_score(df_test_target, df_test_pred)
        print('Training accuracy: ', df_test_acc)
        df_train_pred = nn_model1.predict(df_train)
        df_train_acc = accuracy_score(df_train_target, df_train_pred)
        print('Test accuracy: ', df_train_acc)
        models.plot_confusion_matrix(df_train_target, df_train_pred,names)
        print(metrics.classification_report(df_train_target, df_train_pred, target_names=names))
    else:
        problem_fit = None
        if(model_name.upper() == "TS"):
            # List of coords: https://towardsdatascience.com/solving-travelling-salesperson-problems-with-python-5de7e883d847
            # coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]
            coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3), (4,10), (2,8), (5,6), (3,7), (3, 1), (4, 5),(10,1),(2,11)]
            fitness_coords = mlrose.TravellingSales(coords = coords_list)
            problem_fit = mlrose.TSPOpt(length = len(coords_list), fitness_fn = fitness_coords,
                            maximize=False)
        elif(model_name.upper() == "FF"):
            # state = np.array([0,1,0,1,1,1,1])
            state = np.array([0,1,0,1,1,1,1,0,1,1,1,1,0,1,1,0,1,0,1,1,1,0,1,1,0,1,1,1,1,0,1,1,0,0,0])
            fitness = mlrose.FlipFlop()
            problem_fit = mlrose.DiscreteOpt(length = len(state), fitness_fn = fitness, maximize=True)
        elif(model_name.upper() == "F"):
            state = np.array([0,1,0,1,1,1,1])
            # state = np.array([0,1,0,1,1,1,1,0,1,1,1,1,0,1,1,0,1,0,1,1,1,0,1,1,0,1,1,1,1,0,1,1,0,0,0])
            fitness = mlrose.FourPeaks(t_pct=0.50)
            problem_fit = mlrose.DiscreteOpt(length = len(state), fitness_fn = fitness, maximize=True)

        start_time = time.time()
        # best_state, best_fitness = mlrose.random_hill_climb(problem_fit,  max_attempts=10, max_iters=math.inf, restarts=0, init_state=None)
        best_state, best_fitness, fitness_curve= mlrose.random_hill_climb(problem_fit,  max_attempts=10, max_iters=math.inf, restarts=0, init_state=None, curve = True, random_state = 0)
        print("--- RANDOM HILL CLIMB: %s seconds ---" % (time.time() - start_time))
        print('The best state found is: ', best_state)
        print('The fitness at the best state is: ', best_fitness)
        # print(fitness_curve)
        # plt.figure()
        # plt.title("Fitness Curve")
        # plt.xlabel("Iteration")
        # plt.ylabel("Fitness")
        # plt.plot(fitness_curve, 'o-', color="g")
        # plt.show()

        start_time = time.time()
        # best_state, best_fitness = mlrose.genetic_alg(problem_fit, pop_size=200, mutation_prob=0.1, max_attempts=10,
        #     max_iters=np.inf)
        best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem_fit, pop_size=100, mutation_prob=0.1, max_attempts=10,
            max_iters=np.inf, curve=True, random_state=0)
        print("--- GENETIC ALG FIT: %s seconds ---" % (time.time() - start_time))
        print('The best state found is: ', best_state)
        print('The fitness at the best state is: ', best_fitness)
        # print(fitness_curve)
        # plt.figure()
        # plt.title("Fitness Curve")
        # plt.xlabel("Iteration")
        # plt.ylabel("Fitness")
        # plt.plot(fitness_curve, 'o-', color="g")
        # plt.show()

        start_time = time.time()
        # best_state, best_fitness= mlrose.simulated_annealing(problem_fit)
        best_state, best_fitness, fitness_curve= mlrose.simulated_annealing(problem_fit, random_state = 0, curve=True)
        print("--- SIM ANNEALING FIT: %s seconds ---" % (time.time() - start_time))
        print('The best state found is: ', best_state)
        print('The fitness at the best state is: ', best_fitness)

        # print(fitness_curve)
        # plt.figure()
        # plt.title("Fitness Curve")
        # plt.xlabel("Iteration")
        # plt.ylabel("Fitness")
        # plt.plot(fitness_curve, 'o-', color="g")
        # plt.show()

        start_time = time.time()
        # best_state, best_fitness = mlrose.mimic(problem_fit, pop_size=200, keep_pct=0.2)
        best_state, best_fitness, fitness_curve= mlrose.mimic(problem_fit, pop_size=200, keep_pct=0.2, random_state = 0, curve=True)
        print("--- MIMIC: %s seconds ---" % (time.time() - start_time))
        print('The best state found is: ', best_state)
        print('The fitness at the best state is: ', best_fitness)

        # print(fitness_curve)
        # plt.figure()
        # plt.title("Fitness Curve")
        # plt.xlabel("Iteration")
        # plt.ylabel("Fitness")
        # plt.plot(fitness_curve, 'o-', color="g")
        # plt.show()
