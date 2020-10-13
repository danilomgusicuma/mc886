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
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


#%matplotlib inline

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

def split_last_col(mat):
    x = [line[:-1] for line in mat]
    y = [line[-1] for line in mat]
    return x, y

# x is the features matrix
# y is the prices array
#df = cat_to_num(df)
x, y = split_last_col(df.values)
x = normalize(x)
#2. Split the dataset in train and validation
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=1/5 )

x_train = np.array(x_train)
x_validation = np.array(x_test)
y_train = np.array(y_train)
y_validation = np.array(y_test)

y_train = y_train.reshape((y_train.shape[0], 1))
y_validation = y_test.reshape((y_test.shape[0], 1))

#printing the shape of each resulted array:
print("x_train", x_train)
print("x_train shape", x_train.shape)



# TODO: Linear Regression. Implement your solution. You cannot use scikit-learn libraries.
def random_array_from(x):
    #start the coefficients list randomically
    #generate an array with the size of the number of features
    init_theta = np.random.randn(len(x[0]),1)
    return init_theta

def h_theta(theta, lin_coeff, x):
    # x is a list of features
    # theta is a list of coefficients 
    # h_theta returns the dot product of the theta and x arrays
    predictions = []

    for x_row in x:
        predictions.append(np.dot(theta.T, x_row)+ lin_coeff)    
    return predictions

def complex_h_theta(theta2, theta1, lin_coeff,x):
    predictions = []
    for x_row in x:
        x_row_sq = np.power(x_row, 2)
        predictions.append( np.dot(theta1.T, x_row) + np.dot(theta2.T, x_row_sq) + lin_coeff )
    return predictions

def h_theta_row(theta, lin_coeff, x_row):
    # the same of h_theta but receives only one row of x
    prediction = np.dot(theta.T, x_row)+ lin_coeff
    return prediction

def cost(y, prediction):
    #x is matrix with all the features data
    #prediction is the resultant list of h_theta
    return 1/(2*len(y)) * np.sum((prediction - y)**2)


def derivatives( x, y, prediction):
    m = len(x)
    d_theta = (-1./m)*np.dot(x.T,(y - prediction))
    d_lin = (-1./m)*np.sum((y - prediction))
    return d_theta, d_lin

def complex_derivatives(x, y, prediction):
    m = len(x)
    d_theta2 = (-1./m)*np.dot(np.power(x,2).T,(y - prediction))
    d_theta1 = (-1./m)*np.dot(x.T,(y - prediction))
    d_lin = (-1./m)*np.sum((y - prediction))
    print("wtffffff:", d_theta2, d_theta1, d_lin)
    return d_theta2, d_theta1, d_lin

def batch_GD(x, y, nb_epochs, learning_rate):
    theta = random_array_from(x) * 0.001
    lin_coeff = np.random.randn(1) * 0.001
    for iteration in range(nb_epochs):
        #from the values of theta and the lin_coeff, predict the result
        prediction = h_theta(theta, lin_coeff, x)
        #calculate the cost function
        cost_result = cost(y,prediction)
        #plot the cost
        plt.plot(iteration,cost_result,'go')
        print("iteration:", iteration, "cost:", cost_result)
        #calculate the values for the next iteration
        d_theta, d_lin = derivatives(x,y,prediction)
        #print("d_theta:", d_theta, "d_lin:", d_lin)
        theta = theta - learning_rate * d_theta
        lin_coeff = lin_coeff - learning_rate * d_lin
    return theta, lin_coeff

def stochastic_GD(x, y, nb_epochs, learning_rate):
    theta = random_array_from(x) * 0.001
    lin_coeff = np.random.randn(1) * 0.001
    m = len(x)
    for epoch in range(nb_epochs):
        for iteration in range(m):
            prediction = h_theta_row(theta, lin_coeff, x[iteration])
            x_iteration = x[iteration].reshape((1, x[iteration].shape[0]))
            d_theta, d_lin = derivatives(x_iteration,y[iteration],[prediction])
            new_theta = theta - learning_rate * d_theta
            new_lin_coeff = lin_coeff - learning_rate * d_lin
            theta = new_theta
            lin_coeff = new_lin_coeff
            cost_result = cost(y[iteration],[prediction])
        plt.plot(epoch*10,cost_result,'bo')
        print("iteration:", epoch, "cost:", cost_result)
    return theta, lin_coeff

def mini_batch_GD(x, y, nb_epochs, learning_rate, batch_size):
    theta = random_array_from(x)
    lin_coeff = np.random.randn(1)
    len_x = len(x)
    for epoch in range(nb_epochs):
        for iteration in range(0, len_x, batch_size):
            prediction = h_theta(theta, lin_coeff, x[iteration:iteration+batch_size])
            d_theta, d_lin = derivatives(x[iteration:iteration+batch_size], y[iteration:iteration+batch_size], prediction)
            theta = theta - learning_rate * d_theta
            lin_coeff = lin_coeff - learning_rate * d_lin
            cost_result = cost(y[iteration:iteration+batch_size],prediction)
        print("iteration:", epoch, "cost:", cost_result)
        plt.plot(epoch,cost_result,'ro')
    return theta, lin_coeff

def complex_mini_batch_GD(x, y, nb_epochs, learning_rate, batch_size):
    theta1 = random_array_from(x)
    theta2 = random_array_from(x)
    lin_coeff = np.random.randn(1)
    len_x = len(x)
    for epoch in range(nb_epochs):
        for iteration in range(0, len_x, batch_size):
            prediction = complex_h_theta(theta2, theta1, lin_coeff, x[iteration:iteration+batch_size])
            d_theta2, d_theta1, d_lin = complex_derivatives(x[iteration:iteration+batch_size], y[iteration:iteration+batch_size], prediction)
            theta2 = theta2 - learning_rate * d_theta2
            if d_theta2[0] != d_theta2[0]:
                return
            theta1 = theta1 - learning_rate * d_theta1
            lin_coeff = lin_coeff - learning_rate * d_lin
            cost_result = cost(y[iteration:iteration+batch_size],prediction)
        print("iteration:", epoch, "cost:", cost_result)
        plt.plot(epoch,cost_result,'ro')
    return theta2, theta1, lin_coeff

def normal_equation(x_train, y_train):
    
    x0_norm = np.ones((x_train.shape[0], 1))
    X_norm = np.concatenate((x0_norm, x_train), axis=1)
    # # compute Normal Eq.
    XTX_inv = np.linalg.inv(np.matmul(X_norm.T, X_norm))
    theta_norm = np.matmul(np.matmul(XTX_inv, X_norm.T), y_train)
    return theta_norm

def scikit(x_train, y_train):
    regressor = linear_model.SGDRegressor(loss='squared_loss', penalty='l2', learning_rate = 'constant', eta0 = 0.000001)
    regressor.fit(x_train, np.ravel(y_train))
    prediction = regressor.predict(x_train)
    print("scikit cost:", cost(y_train, np.array(prediction).reshape(prediction.shape[0],1)))
    print("prediction:", np.array(prediction).reshape(prediction.shape[0],1))
    print("values:", y_train)
    
    return regressor.coef_


# results1 = mini_batch_GD(x_train, y_train, 100, 0.001, 10)
# results2 = stochastic_GD(x_train, y_train, 10, 0.001)
# results3 = batch_GD(x_train, y_train, 100, 0.01)
# results4 = normal_equation(x_train, y_train)
# results5 = scikit(x_train, y_train)
results6 = complex_mini_batch_GD(x_train, y_train, 100, 0.01, 10)

plt.show()
# print("mini batch:",results1)
# print("stochastic_GD:",results2)
# print("batch_GD:",results3)
# print("normal:", results4)
# print("scikit:", results5)
print("complex mini batch:",results6)





# TODO: Linear Regression. Implement your solution with sklearn.linear_model.SGDRegressor.
# TODO: Complex model. Implement your solution. You cannot use scikit-learn libraries.
# TODO: Plot the cost function vs. number of iterations in the training set.
# TODO: Gradient Descent (GD) with different learning rates. Implement your solution. You cannot use scikit-learn libraries.
# TODO: Compare the GD-based solutions (e.g., Batch, SGD, Mini-batch) with Normal Equation. Implement your solution. You cannot use scikit-learn libraries.

