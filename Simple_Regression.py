import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)

df = pd.read_csv('headbrain.csv')
X = df['Head Size(cm^3)'].values
Y = df['Brain Weight(grams)'].values

n = len(X)
def mean(m):
    return np.mean(m)

def LinearRegression(X,Y):
    num = 0
    den = 0
    for i in range(n):
        num = (X[i] - mean(X)) * (Y[i] - mean(Y))
        den = (X[i] - mean(X))**2
        b1 = num/den
        b0 = mean(Y) - b1*mean(X)
        return (b0, b1)


def Rmse(X,Y):
    b0, b1 = LinearRegression(X,Y)
    for i in range(n):
        Y_predict = b0 + b1*X[i]
        mse += (Y_predict - Y[i])**2
    rmse = np.sqrt(mse/n)
    print("Root Mean Square Error: ",rmse)

def score(X, Y):
    num = 0
    den = 0
    b0, b1 = LinearRegression(X,Y)
    for i in range(n):
        Y_predict = b0 + b1*X[i]
        num = (Y[i] - Y_predict)**2
        den = (Y[i] - mean(Y))**2
        s = 1 - (num/den)
    return s

def predict(x):
    b0, b1 = LinearRegression(X,Y)
    return b0 + b1*x
    
b0,b1 = LinearRegression(X,Y)
accuracy = score(X,Y)
print(accuracy)
print("Enter Head Size in cm3")
k = int(input())
print("Brain Weight:", predict(k))
'''
#Plotting
max_x = np.max(X) + 100
min_x = np.min(X) - 100

# Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x

# Ploting Line
plt.plot(x, y, color='#58b970', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')

plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()


'''
        
        
        
    
