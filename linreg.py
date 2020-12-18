from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
from sympy import Symbol, Derivative


def firstlinregmodel(x, y, epoch, lr, x_test):
    """
    lr = learning rate
    epoch = num of batches
    x : array-like, shape = [n_samples, n_features]
        Training samples
    y : array-like, shape = [n_samples, n_target_values]
        Target values
    both x and y are series. x is assumed to be just one series

    ****each datapoint has a vertical and horizontal distance
    """
    # y = mx + c

    # pick a random line
    # which is to say pick random numbers for m and c
    m = 0
    c = 0
    sum_sq_residuals = []
    slope = []
    intercept = []
    # make the values into data points with vertical and horizontal distance
    data = dict(zip(x, y))

    for _ in range(epoch):
        # to pick a random point
        x_random, y_random = random.choice(list(data.items()))

        # if the point is above the line and to the right of the y axis
        if (m*x_random)+c-y_random > 0 and x_random > 0:
            m += lr
            c += lr
        # if the point is above the line and to the left of the y axis
        elif (m*x_random)+c-y_random > 0 and x_random < 0:
            m -= lr
            c += lr
        # if the point is below the line and to the left of the y axis
        elif (m*x_random)+c-y_random < 0 and x_random < 0:
            m -= lr
            c -= lr
        # if the point is below the line and to the right of the y axis
        elif (m*x_random)+c-y_random < 0 and x_random > 0:
            m += lr
            c -= lr
        sum_sq_residuals.append(sum((m*x+c - y)**2))
        slope.append(m)
        intercept.append(c)
    # end of training --> we have our m,c
    # evaluations and plots
    y_train_pred = m*x+c
    len_x = x.shape[0]
    # 'map' train pred to actual
    # output_pair = dict(zip(y_train_pred, y))
    # print(" printing y_train_predict vs y")
    # print(output_pair)
    # calculate RMSE
    # res = []
    # for pred, act in output_pair.items():
    #     residual = (pred - act)**2
    #     res.append(residual)
    # rmse = np.sqrt(sum(res)/len_x)
    # print(" rmse: ", rmse)

    # mean squared error
    mse = np.sum((y_train_pred - y)**2)
    # root mean squared error
    rmse = np.sqrt(mse/len_x)

    # R2 score
    sst = sum((y-np.mean(y))**2)
    ssr = sum((y_train_pred - y)**2)
    r2 = 1 - (ssr/sst)

    # printing values
    print('Slope:', c)
    print('Intercept:', m)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)
    # PLOTS
    # data points
    plt.scatter(x, y, s=10)
    plt.xlabel('math score')  # x
    plt.ylabel('reading score')  # y

    # predicted values
    plt.plot(x, y_train_pred, color='r')
    plt.show()
    # predict on test data
    y_test_pred = m*x_test+c
    print(' y_test_predict: ', y_test_pred)
    plt.plot(slope, sum_sq_residuals, color='green')
    plt.xlabel('slope')  # x
    plt.ylabel('sum of squared residuals')  # y
    plt.show()
    return


def absolutetrick(x, y, epoch, lr):
    # ABSOLUTE TRICK
    # THIS IS FOR THE 2ND ALGO
    # pick a random line
    # which is to say pick random numbers for m and c
    m = 0
    c = 0
    # make the values into data points with vertical and horizontal distance
    data = dict(zip(x, y))

    for _ in range(epoch):
        # to pick a random point
        x_random, y_random = random.choice(list(data.items()))
        # measure the vertical and horizontal distances of each random point
        # vertical dist = from line to the point
        # horizontal dist = from y-axis to point
        h_dist = x_random

        # if point is above the line
        if (m*x_random)+c-y_random > 0:
            m += lr*h_dist
            c += lr
        else:
            m -= lr*h_dist
            c -= lr

    # end of training --> we have our m,c
    # evaluations and plots
    y_train_pred = m*x+c
    len_x = x.shape[0]
    # 'map' train pred to actual
    # output_pair = dict(zip(y_train_pred, y))
    # print(" printing y_train_predict vs y")
    # print(output_pair)
    # calculate RMSE
    # res = []
    # for pred, act in output_pair.items():
    #     residual = (pred - act)**2
    #     res.append(residual)
    # rmse = np.sqrt(sum(res)/len_x)
    # print(" rmse: ", rmse)

    # mean squared error
    mse = np.sum((y_train_pred - y)**2)
    # root mean squared error
    rmse = np.sqrt(mse/len_x)

    # R2 score
    sst = sum((y-np.mean(y))**2)
    ssr = sum((y_train_pred - y)**2)
    r2 = 1 - (ssr/sst)

    # printing values
    print('Slope:', c)
    print('Intercept:', m)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)
    # PLOTS
    # data points
    plt.scatter(x, y, s=10)
    plt.xlabel('math score')  # x
    plt.ylabel('reading score')  # y

    # predicted values
    plt.plot(x, y_train_pred, color='r')
    plt.show()

    return


def squaretrick(x, y, epoch, lr):
    # SQUARE TRICK
    # THIS IS FOR THE 2ND ALGO
    # pick a random line
    # which is to say pick random numbers for m and c
    m = 0
    c = 0
    # make the values into data points with vertical and horizontal distance
    data = dict(zip(x, y))

    for _ in range(epoch):
        # to pick a random point
        x_random, y_random = random.choice(list(data.items()))
        # measure the vertical and horizontal distances of each random point
        # vertical dist = from line to the point
        # horizontal dist = from y-axis to point
        h_dist = x_random
        v_dist = (m*x_random)+c-y_random
        m += lr*h_dist*v_dist
        c += lr*v_dist

    # end of training --> we have our m,c
    # evaluations and plots
    y_train_pred = m*x+c
    len_x = x.shape[0]
    # 'map' train pred to actual
    # output_pair = dict(zip(y_train_pred, y))
    # print(" printing y_train_predict vs y")
    # print(output_pair)
    # calculate RMSE
    # res = []
    # for pred, act in output_pair.items():
    #     residual = (pred - act)**2
    #     res.append(residual)
    # rmse = np.sqrt(sum(res)/len_x)
    # print(" rmse: ", rmse)

    # mean squared error
    mse = np.sum((y_train_pred - y)**2)
    # root mean squared error
    rmse = np.sqrt(mse/len_x)

    # R2 score
    sst = sum((y-np.mean(y))**2)
    ssr = sum((y_train_pred - y)**2)
    r2 = 1 - (ssr/sst)

    # printing values
    print('Slope:', c)
    print('Intercept:', m)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)
    # PLOTS
    # data points
    plt.scatter(x, y, s=10)
    plt.xlabel('math score')  # x
    plt.ylabel('reading score')  # y

    # predicted values
    plt.plot(x, y_train_pred, color='r')
    plt.show()
    # predict on test data
    y_test_pred = m*x_test+c
    print(' y_test_predict: ', y_test_pred)

    return


def squareError():

    return


data = pd.read_csv('stuperf.csv')
train_data = data[0:700]
test_data = data[700:]
y = train_data['math score']
# y = y.values.reshape((700, 1))
x = train_data['reading score']
# x = x.values.reshape((700, 1))
x_test = test_data['reading score']

print(firstlinregmodel(x, y, 1000, 0.001, x_test))
print(absolutetrick(x, y, 1000, 0.00001))
#print(squaretrick(x, y, 1000, 0.000001))
