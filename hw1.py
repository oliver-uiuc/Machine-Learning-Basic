import numpy as np
import hw1_utils as utils
import matplotlib.pyplot as plt

def linear_gd(X,Y,lrate=0.1,num_iter=1000):
    samples, features = X.shape
    X_aug = np.hstack((np.ones((samples, 1)), X)) 
    w = np.zeros(features + 1)
    for i in range(num_iter):
        grad = (1/samples) * X_aug.T.dot(X_aug.dot(w) - Y) 
        w -= lrate * grad
    return w

def linear_normal(X,Y):
    samples, _ = X.shape
    X_aug = np.hstack((np.ones((samples, 1)), X))
    w = np.linalg.inv(X_aug.T.dot(X_aug)).dot(X_aug.T).dot(Y)
    return w


def plot_linear():
    
    X, Y = utils.load_reg_data()  
    w = linear_normal(X, Y)  
    n = X.shape[0]
    X_aug = np.hstack((np.ones((n, 1)), X))
    Y_pred = X_aug.dot(w)
    plt.scatter(X, Y, color='blue', label='Data Points') 
    sort_order = np.argsort(X[:, 0])
    plt.plot(X[sort_order, 0], Y_pred[sort_order], color='red', label='Regression Line')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_linear()    