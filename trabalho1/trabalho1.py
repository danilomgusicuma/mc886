import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

import normalizeUtils as utils

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

#%matplotlib inline

test_dot = np.dot(np.array([[1],[2]]), np.array([[1,1,3]]))
print('test-dot:', test_dot)

df=pd.read_csv("diamonds-train.csv")

#1. Normalizing the dataset features
def normalize(values): # values is the matrix with all the data
    #normalizing features values to [0,10]
    rows = [ [ 0 for i in range(9) ] for j in values ]
    for i in range(len(values)):
        rows[i][0] = (values[i][0] - 0.2)/0.481
        rows[i][1] = utils.cutNumber(values[i][1])
        rows[i][2] = utils.colorNumber(values[i][2])
        rows[i][3] = utils.clarityNumber(values[i][3])
        rows[i][4] = values[i][4]/1.074
        rows[i][5] = values[i][5]/5.89
        rows[i][6] = values[i][6]/3.18
        rows[i][7] = (values[i][7] - 43)/3.6
        rows[i][8] = (values[i][8] - 43)/5.2
    return rows

# x is the features matrix
# y is the prices array
x = normalize(df.values)
y = []
for i in df.values:
    y.append(i[9])

#2. Split the dataset in train and validation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/5 )

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

y_train = y_train.reshape((y_train.shape[0], 1))
y_test = y_test.reshape((y_test.shape[0], 1))

#printing the shape of each resulted array:
print("x_train.shape", x_train)
print("x_test.shape", x_test.shape)
print("y_train.shape", y_train.shape)
print("y_test.shape", y_test.shape)



# TODO: Linear Regression. Implement your solution. You cannot use scikit-learn libraries.

def h_theta(theta, x):
    # x is a list of features
    # theta is a list of coefficients 
    # h_theta returns the dot product of the theta and x arrays
    return np.sum(np.multiply(theta, x))

test = h_theta([1,2,3], [2,3,4])

def cost(theta, x, y):
    #x is matrix with all the features data
    aux = 0
    for i in range(len(x)):
        aux += (h_theta(theta, x[i]) - y[i])**2
    return 1/(2*len(x)) * aux






# TODO: Linear Regression. Implement your solution with sklearn.linear_model.SGDRegressor.
# TODO: Complex model. Implement your solution. You cannot use scikit-learn libraries.
# TODO: Plot the cost function vs. number of iterations in the training set.
# TODO: Gradient Descent (GD) with different learning rates. Implement your solution. You cannot use scikit-learn libraries.
# TODO: Compare the GD-based solutions (e.g., Batch, SGD, Mini-batch) with Normal Equation. Implement your solution. You cannot use scikit-learn libraries.

