import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

df=pd.read_csv("diamonds-train.csv")

def normalize(values):
    rows = [ [ 0 for i in range(9) ] for j in values ]
    for i in range(len(values)):
        rows[i][0] = (values[i][0] - 0.2)/0.481
        rows[i][1] = cutNumber(values[i][1])
        rows[i][2] = colorNumber(values[i][2])
        rows[i][3] = clarityNumber(values[i][3])
        rows[i][4] = values[i][4]/1.074
        rows[i][5] = values[i][5]/5.89
        rows[i][6] = values[i][6]/3.18
        rows[i][7] = (values[i][7] - 43)/3.6
        rows[i][8] = (values[i][8] - 43)/5.2
    return rows

# TODO: Linear Regression. Implement your solution. You cannot use scikit-learn libraries.

def h_theta(theta, x):
    return np.sum(np.multiply(theta, x))

test = h_theta([1,2,3], [2,3,4])

def cutNumber(cut):
    if cut == "Fair":
        return 0
    elif cut == "Good":
        return 2.5
    elif cut == "Very Good":
        return 5
    elif cut == "Premium":
        return 7.5
    elif cut == "Ideal":
        return 10

def colorNumber(color):
    if color == "J":
        return 0
    elif color == "I":
        return 5/3
    elif color == "H":
        return 10/3
    elif color == "G":
        return 5
    elif color == "F":
        return 20/3
    elif color == "E":
        return 25/3
    elif color == "D":
        return 10

def clarityNumber(clarity):
    if clarity == "I1":
        return 0
    elif clarity == "SI2":
        return 10/7
    elif clarity == "SI1":
        return 20/7
    elif clarity == "VS2":
        return 30/7
    elif clarity == "VS1":
        return 40/7
    elif clarity == "VVS2":
        return 50/7
    elif clarity == "VVS1":
        return 60/7
    elif clarity == "IF":
        return 10

def cost(theta, x, y):
    aux = 0
    for i in x:
        aux += (h_theta(theta, i) - y[i])**2
    return 1/(2*len(x)) * aux

x = normalize(df.values)
y = []
for i in df.values:
    y.append(i[9])

# TODO: Linear Regression. Implement your solution with sklearn.linear_model.SGDRegressor.
# TODO: Complex model. Implement your solution. You cannot use scikit-learn libraries.
# TODO: Plot the cost function vs. number of iterations in the training set.
# TODO: Gradient Descent (GD) with different learning rates. Implement your solution. You cannot use scikit-learn libraries.
# TODO: Compare the GD-based solutions (e.g., Batch, SGD, Mini-batch) with Normal Equation. Implement your solution. You cannot use scikit-learn libraries.

print(cost([0,0,0,0,0,0,0,0,0], x, y))
