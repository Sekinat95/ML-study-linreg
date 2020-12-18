from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
from sympy import Symbol, Derivative


def mygradientdescent(X, Y, epoch, L):
    m = 0
    c = 0
    n = float(len(X))  # Number of elements in X
    # Performing Gradient Descent

    for _ in range(epoch):
        Y_pred = m*X + c  # The current predicted value of Y
        D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
        D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
        m = m - L * D_m  # Update m
        c = c - L * D_c  # Update c
        #print(m, c)
    Y_pred = m*X + c

    plt.scatter(X, Y, s=10)
    plt.plot(x, Y_pred, color='r')  # predicted
    plt.xlabel('reading score')  # x
    plt.ylabel('math score')  # y
    plt.show()
    return


data = pd.read_csv('stuperf.csv')
train_data = data[0:700]
test_data = data[700:]
y = train_data['math score']
# y = y.values.reshape((700, 1))
x = train_data['reading score']
# x = x.values.reshape((700, 1))
x_test = test_data['reading score']

mygradientdescent(x, y, 1000, 0.00001)
