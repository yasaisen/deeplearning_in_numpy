
import numpy as np
import matplotlib.pyplot as plt

class activation_functions:
    def step_function(x):
        return np.where(x<=0, 0, 1)# if([1]){[2]}else{[3]}

    def sigmoid_function(x):
        return 1/(1+np.exp(-x))

    def tanh_function(x):
        return np.tanh(x)

    def relu_function(x):
        return np.where(x <= 0, 0, x)# if([1]){[2]}else{[3]}

    def leaky_relu_function(x):
        return np.where(x <= 0, 0.01*x, x)# if([1]){[2]}else{[3]}

    def identity_function(x):
        return x

    def softmax_function(x):
        return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)# [0.09003057 0.24472847 0.66524096]



# x = np.linspace(-5, 5)



# plt.plot(x, activation_functions.step_function(x))
# plt.title('step_function')
# plt.figure()

# plt.plot(x, activation_functions.sigmoid_function(x))
# plt.title('sigmoid_function')
# plt.figure()

# plt.plot(x, activation_functions.tanh_function(x))
# plt.title('tanh_function')
# plt.figure()

# plt.plot(x, activation_functions.relu_function(x))
# plt.title('relu_function')
# plt.figure()

# plt.plot(x, activation_functions.leaky_relu_function(x))
# plt.title('leaky_relu_function')
# plt.figure()


# plt.plot(x, activation_functions.identity_function(x))
# plt.title('identity_function')
# plt.show()

# print(activation_functions.softmax_function(np.array([1,2,3])))

