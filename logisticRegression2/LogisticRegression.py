__author__ = 'tend'


import numpy as np
from pylab import scatter,show,legend,xlabel,ylabel

data = np.loadtxt('testSet.txt', delimiter=',')

X = data[:,0:2]
y = data[:,2]

pos = np.where(y == 1) #select
neg = np.where(y == 0)

scatter(X[pos,0],X[pos,1],marker='o',c='b')
scatter(X[neg,0],X[neg,1],marker='x',c='r')
xlabel('Featrue1/Exam 1 score')
ylabel('Featrue2/Exam 2 score')
legend(['Fail','Pass'])
show()



def sigmoid(X):
    '''''compute sigmoid function'''
    den = 1.0 + np.exp(-1.0 * X)
    gz = 1.0/den
    return gz

#
def compute_cost(theta,X,y):
    '''''computes cost given predicted and actual values'''
    m = X.shape[0] #number of training examples
    theta = np.reshape(theta,(len(theta),1))

    h = sigmoid(X.dot(theta))

    J = (1./m) * ( -np.transpose(y).dot(np.log(h)) - np.transpose(1 - y).dot(np.log(1 - h)))

    return J


def compute_grad(theta,X,y):
    '''''compute gradient'''
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1,1)))

    grad = (1./m)* np.transpose(X).dot(h - y)

    return (grad.flatten())




initial_theta = np.zeros(X.shape[1])
cost = compute_cost(initial_theta,X,y)
grad = compute_grad(initial_theta,X,y)
print('Cost: \n', cost)
print('Grad: \n', grad)

