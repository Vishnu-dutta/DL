import math as mt

'''
Its often used in binary classification problems, where the output of the model needs to 
a probability between 0 and 1 
'''


def sigmoid(x):
    a = 1 / (1 + mt.exp(-x))
    return a


'''
Like sigmoid function but maps any real-values number to a value between -1 and 1. Its often 
used as an activation function hidden layer
'''


def tanh(x):
    a = (mt.exp(x) - mt.exp(-x)) / (mt.exp(x) + mt.exp(-x))
    return a


'''
It maps any negative input to 0 and any positive input to itself. Its often used in neural networks.
It works well due to simplicity and computational efficiency
'''


def ReLU(x):
    return max(0, x)


'''
The Leaky ReLU is similar to ReLU function but allows a small non-zero gradient when the input is negative.
Its often used in neural networks to avoid the 'dying ReLU' problem, where neurons become unresponsive if their 
output is always 0 
'''


def leaky_relu(x):
    return max(0.1 * x, x)


print('sigmoid for 50: ', sigmoid(50))
print('sigmoid for -50: ', sigmoid(-50))
print('tanh for 50: ', tanh(50))
print('tanh for -50: ', tanh(-50))
print('relu for 50: ', ReLU(50))
print('relu for -50', ReLU(-50))
print('Leaky relu for 50:', leaky_relu(50))
print('Leaky relu for -50:', leaky_relu(-50))