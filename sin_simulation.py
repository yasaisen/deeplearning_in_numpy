
from itertools import count
from activation import activation_functions
from loss import loss_functions
import numpy as np
import matplotlib.pyplot as plt

# —準備輸入與正解—
input_data = np.arange(0, np.pi*2, 0.1) # use "arange" function to make 0.0 to pi*2 numbers with 0.1 betweened, is a segmented horizontal line
correct_data = np.sin(input_data)       # labels, make segmented horizontal line to sin with y-axis
input_data = (input_data-np.pi)/np.pi   # make it into -1.0～1.0 no range



# —各設定值—
neuron_in  = 1   # number of neuron on input layer
neuron_mid = 3   # number of neuron on middle layer
neuron_out = 1   # number of neuron on output layer

wb_width = 0.01  # let weights and bias smaller
eta = 0.1        # Learning_rate
epoch = 2001     # epoch
interval = 200   # epoch no increase rate
                 # batch_size

 # use Gradient Descent to fix weights
 # backward propagation
 # Optimization Algorithm  : Stochastic_Gradient_Descent, Momentum, Adagrad, RMSProp, Adam



# -- 中間層 --
class MiddleLayer:
    def __init__(self, neuron_upper, neuron_current):  # 初期設定
        self.w1 = wb_width * np.random.randn(neuron_upper, neuron_current)  # use the number of neuron_upper and neuron_current to setup kokono layer no weights (neuron_upper * neuron_current)(matrix)
        self.b1 = wb_width * np.random.randn(neuron_current)  # use the number of neuron_current to setup kokono layer no bias (1 * neuron_current)(vector)

    def forward(self, x):  # forward_propagation
        self.x = x  # single value from input_data, save for backward_propagation
        u1 = np.dot(x, self.w1) + self.b1  # neuron is here
        # self.y1 = 1/(1+np.exp(-u1))  # sigmoid
        self.y1 = activation_functions.sigmoid_function(u1)
    
    def backward(self, grad_y1):  # backward_propagation
        delta1 = grad_y1 * (1-self.y1)*self.y1  # sigmoid 函數的微分################################
        
        self.grad_w1 = np.dot(self.x.T, delta1)  # the gradient for weight, size : ( * )
        self.grad_b1 = np.sum(delta1, axis=0)    # the gradient for bias, size : ( * )

        # have no front neuron need to update, so no gradients compute
                
    def update(self, eta):  # update weights and bias
        self.w1 -= eta * self.grad_w1  # Stochastic_Gradient_Descent######################################################################
        self.b1 -= eta * self.grad_b1  # Stochastic_Gradient_Descent

# -- 輸出層 --
class OutputLayer:
    def __init__(self, neuron_upper, neuron_current):  # 初期設定
        self.w2 = wb_width * np.random.randn(neuron_upper, neuron_current)  # use the number of neuron_upper and neuron_current to setup kokono layer no weights (neuron_upper * neuron_current)(matrix)
        self.b2 = wb_width * np.random.randn(neuron_current)  # use the number of neuron_current to setup kokono layer no bias (1 * neuron_current)(vector)
    
    def forward(self, y1):  # forward_propagation
        self.y1 = y1  # single value from upper layer, save for backward_propagation
        u2 = np.dot(y1, self.w2) + self.b2  # neuron is here
        # self.y2 = u2  # identity
        self.y2 = activation_functions.identity_function(u2)
    
    def backward(self, t):  # backward_propagation
        delta2 = self.y2 - t  # delta mean_square_error(5-25, 5-52)##########################################
                
        self.grad_w2 = np.dot(self.y1.T, delta2)  # the gradient for weight, size : ( * )
        self.grad_b2 = np.sum(delta2, axis=0)     # the gradient for bias, size : ( * )
        
        self.grad_y1 = np.dot(delta2, self.w2.T)  # the gradient for input from the arrow of front neuron to here, size : ( * )

    def update(self, eta):  # update weights and bias
        self.w2 -= eta * self.grad_w2  # Stochastic_Gradient_Descent
        self.b2 -= eta * self.grad_b2  # Stochastic_Gradient_Descent


# -- 各層的初始化 --
middle_layer = MiddleLayer(neuron_in, neuron_mid)  # input neuron number, 1 to 3
output_layer = OutputLayer(neuron_mid, neuron_out) # input neuron number, 3 to 1


number_of_data = len(correct_data)  # data no ryo

counter = 0

# -- 學習 --
for i in range(epoch):

      # randomize index
    index_random = np.arange(number_of_data)  # index making, 63
    np.random.shuffle(index_random)           # randomize index 
    
      # results container initialization
    total_error = 0  # +=ed and printed ,so here is initialization
    plot_x = []      # loged and printed ,so here is initialization
    plot_y2 = []     # loged and printed ,so here is initialization
    
    for idx in index_random:  # use sagi tsukude no randomize index to let every epoch no input is random
        
        x = input_data[idx:idx+1].reshape(1, 1)  # input_data no single value input and reshape it into 2Darray
        
          # forward_propagation
        middle_layer.forward(x)  # input single input_data value to do forward_propagation
        output_layer.forward(middle_layer.y1)  # input result from upper layer


        t = correct_data[idx:idx+1].reshape(1, 1)  # correct_data no single value input and reshape it into 2Darray

          # backward_propagation
        output_layer.backward(t)  # input single correct_data value to do backward_propagation
        middle_layer.backward(output_layer.grad_y1)  # input result from lower layer


          # update weights and bias
        middle_layer.update(eta)  # input learning_rate
        output_layer.update(eta)  # input learning_rate
        

        if i%interval == 0:  # if epoch run interval times, called 63 * 11 times, ***first time keep in
            
            y2 = output_layer.y2.reshape(-1)  # reshape 2Darray to 1Darray
            
            
              # compute loss
            # total_error += 1.0/2.0*np.sum(np.square(y2 - t))  # MSE
            total_error += loss_functions.mean_square_error(y2, t)  # compute loss between last_output and last_lable
            

              # log outputed point to draw
            plot_x.append(x)    # because of the random index, we need to log current input_data
            plot_y2.append(y2)  # log the output ans
            
    if i%interval == 0:  # called 11 times
        
          # print
        plt.plot(input_data, correct_data, linestyle="dashed")
        plt.scatter(plot_x, plot_y2, marker="+")  # use sagi loged
        plt.show()
        
        # 顯示epoch 次數與誤差
        print("Epoch:" + str(i) + "/" + str(epoch), "Error:" + str(total_error/number_of_data))

