
from activation import activation_functions
import numpy as np
import matplotlib.pyplot as plt

# x_0 = np.arange(-1.0, 1.0, 0.2) # use "arange" function to make -1.0 to 1.0 numbers with 0.2 betweened, zenbu no ryo ha 10 go
# x_1 = np.arange(-1.0, 3.0, 0.2) # use "arange" function to make -1.0 to 1.0 numbers with 0.2 betweened, zenbu no ryo ha 20 go
x_0 = np.arange(-1.0, 1.0, 0.1) # make -1.0 to 1.0 numbers with 0.1 betweened, zenbu no ryo ha 20 go
x_1 = np.arange(-1.0, 2.0, 0.1) # make -1.0 to 1.0 numbers with 0.1 betweened, zenbu no ryo ha 20 go

print(x_0)
print(x_1)


x_0_num = len(x_0) # because ue zenbu no ryo ha 10 go
x_1_num = len(x_1) # because ue zenbu no ryo ha 20 go

# w_x_0 = 2.5 # first arrow no weight
# w_x_1 = 3.5 # second arrow no weight
# w_im = np.array([[4.0,4.0],
#                  [4.0,4.0]]) # 2 neurons to 2 neurons they have 2 + 2 (=4)arrows
# w_mo = np.array([[ 1.0],
#                  [-1.0]]) # 2 neurons to 1 neurons they have 1 + 1 (=2)arrows
w_im = np.array([[1.0,2.0],
                 [2.0,3.0]]) # 2 neurons to 2 neurons they have 2 + 2 (=4)arrows
w_mo = np.array([[-1.0,1.0],
                 [1.0,-1.0]]) # 2 neurons to 2 neurons they have 2 + 2 (=4)arrows

# bias = 0.1 # kono neuron no bias
# b_im = np.array([3.0,-3.0]) # front 2 neurons no bias
# b_mo = np.array([0.1]) # saigo no neuron no bias
b_im = np.array([0.3,-0.3]) # 2 neurons no bias
b_mo = np.array([0.4, 0.1]) # 2 neurons no bias

# Z = np.zeros(x_0_num * x_1_num) # create a all-null nparray to catch the result at last
Z = np.zeros((x_0_num * x_1_num, 2)) # because all neuron output 2 values, to catch all of them, we use (x_0_num * x_1_num, 2)


def middle_layer(x, w, b): # middle_layer input ([x_0[i], x_1[j]]), w_im, b_im
    u = np.dot(x, w) + b # neuron is here
    print(u)
    # return 1/(1+np.exp(-u)) # sigmoid
    return activation_functions.sigmoid_function(u) # return 1Darray with 2 values


def output_layer(x, w, b): # output_layer input ([middle_layer(x_0[i]), middle_layer(x_1[j])]), w_im, b_im
    u = np.dot(x, w) + b # neuron is here
    print(u)
    # return u # identity
    # return activation_functions.identity_function(u) # return 1Darray with 1 values
    return activation_functions.softmax_function(u) # return 1Darray with 2 values



for i in range(x_0_num):     # kokono 10 ha x_0 zenbu no ryo (10 go)
    for j in range(x_1_num): # kokono 10 ha x_1 zenbu no ryo (20 go)
        # u = x_0[i] * w_x_0 + x_1[j] * w_x_1 + bias # neuron is here
        # # y = 1/(1+np.exp(-u)) # sigmoid
        # y = activation_functions.sigmoid_function(u)
        # Z[i * x_1_num + j] = y # to put result in a 1D array

        inp = np.array([x_0[i], x_1[j]]) # to make "dot" operator easily
        mid = middle_layer(inp, w_im, b_im) 
        out = output_layer(mid, w_mo, b_mo) 
        # Z[i* x_1_num +j] = out[0] # to put [0] is because that the "out" from up is an 1Darray with just one value, we just need that.
        Z[i* x_1_num +j] = out # 2 to 2 inputs, all 400 go

        
print(Z)

# plt.imshow(Z.reshape(x_0_num,x_1_num), "gray", vmin = 0.0, vmax =1.0)  # use "reshape" function to parse 1Darray to 2Darray
# plt.colorbar()


#透過 Z 的結果將 400 組 xy 組合分類，並畫成圖

plus_x = []
plus_y = []
circle_x = []
circle_y = []

for i in range(x_0_num): # visit all Z
    for j in range(x_1_num): # visit all Z
        if Z[i* x_1_num +j][0] > Z[i* x_1_num +j][1]: # compare the two probabilities
            plus_x.append(x_0[i]) # (x_0[i], x_1[j]) more likely to be (+)
            plus_y.append(x_1[j])
        else:
            circle_x.append(x_0[i]) # (x_0[i], x_1[j]) more likely to be (o)
            circle_y.append(x_1[j])


plt.scatter(plus_x, plus_y, marker="+")
plt.scatter(circle_x, circle_y, marker="o")


plt.show()
