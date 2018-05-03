import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['figure.figsize'] = (20.0, 10.0)

df = pd.read_csv('student.csv')
math = df['Math'].values
Reading = df['Reading'].values
Writing = df['Writing'].values

rows,col = df.shape
Y = np.array(Writing)
m = len(Y)
x0 = np.ones(m)
X = np.array([x0, math, Reading]).T
B = np.zeros((col,))



alpha = 0.0001

def mean(m):
    return np.mean(m)

def cost_function(X,Y,B):
    h = X.dot(B)
    loss = h - Y
    cost = np.sum((loss)**2)/(2*m)
    return cost

def gradient_descent(X,Y,B,alpha,iterations):
    cost_history = [0]*iterations
    for iteration in range(iterations):
        loss = X.dot(B)-Y
        gradient = X.T.dot(loss)/m
        B = B - alpha*gradient
        cost = cost_function(X,Y,B)
        cost_history[iteration] = cost
    return B, cost_history


def Rmse(Y,Y_predict):
    mse += (Y_predict - Y)**2
    rmse = sum(np.sqrt(mse/m))
    print("Root Mean Square Error: ",rmse)

def score(Y, Y_predict):
    num = sum((Y - Y_predict)**2)
    den = sum((Y - mean(Y))**2)
    s = 1 - (num/den)
    return s

def predict(myX):
    return myX.dot(Bnew)

print("initial cost: ", cost_function(X,Y,B))
Bnew, cost_history = gradient_descent(X,Y,B,alpha,10000)
print("new B: ", Bnew)
print("cost after gradientdescent: ",cost_history[-1])

Y_predict = X.dot(Bnew)
accuracy = score(Y,Y_predict)
print("accuracy: ",accuracy)

myX = np.array([1,70,85]).T
print("prediction: ",predict(myX))


'''
# Ploting the scores as scatter plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(math, Reading, Writing, color='#ef1234')
plt.show()
'''
