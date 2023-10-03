
from activation import activation_functions
import numpy as np
import matplotlib.pyplot as plt

x_0 = np.arange(-1.0, 1.0, 0.2) # use "arange" function to make -1.0 to 1.0 numbers with 0.2 betweened, zenbu no ryo ha 10 go
x_1 = np.arange(-1.0, 3.0, 0.2) # use "arange" function to make -1.0 to 3.0 numbers with 0.2 betweened, zenbu no ryo ha 20 go

print(x_0)
print(x_1)


x_0_num = len(x_0) # because ue zenbu no ryo ha 10 go
x_1_num = len(x_1) # because ue zenbu no ryo ha 20 go

w_x_0 = 2.5 # first arrow no weight
w_x_1 = 1.5 # second arrow no weight

bias = 0.1 # kono neuron no bias


Z = np.zeros(x_0_num * x_1_num) # create a all-null nparray to catch the result at last

for i in range(x_0_num):     # kokono 10 ha x_0 zenbu no ryo (10 go)
    for j in range(x_1_num): # kokono 10 ha x_1 zenbu no ryo (20 go)
        u = x_0[i] * w_x_0 + x_1[j] * w_x_1 + bias # neuron is here
        # y = 1/(1+np.exp(-u)) # sigmoid
        y = activation_functions.sigmoid_function(u)
        Z[i * x_1_num + j] = y # to put result in a 1D array
print(Z)

plt.imshow(Z.reshape(x_0_num,x_1_num), "gray", vmin = 0.0, vmax =1.0)  # use "reshape" function to parse 1Darray to 2Darray
plt.colorbar()  
plt.show()
