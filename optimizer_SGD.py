
from activation import activation_functions
from loss import loss_functions
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# -- 載入 Iris 資料集 --
iris_data = datasets.load_iris()

input_data = iris_data.data  # (4 *150)2Darray
correct = iris_data.target   # (150)1Darray
n_data = len(correct)        # data no ryo (150)

# -- 將訓練樣本做正規化 (Normalization) 處理 --
input_data = (input_data - np.average(input_data, axis=0)) / np.std(input_data, axis=0)

# -- 將標籤 (正確答案) 做 one-hot 編碼 --
correct_data = np.zeros((n_data, 3))   # 3 type of lable (0, 1, 2)
for i in range(n_data):
    correct_data[i, correct[i]] = 1.0  # (3 * 150)2Darray

# -- 將資料集拆分為「訓練資料集」與「測試資料集」 --
index = np.arange(n_data)          # index making, 150
index_train = index[index%2 == 0]  # index of train
index_test = index[index%2 != 0]   # index of test

input_train = input_data[index_train, :]
correct_train = correct_data[index_train, :]

input_test = input_data[index_test, :]
correct_test = correct_data[index_test, :]

n_train = input_train.shape[0]  # data no ryo (75)
n_test = input_test.shape[0]    # data no ryo (75)



# -- 各個設定值 --
n_in  = 4       # number of neuron on input layer
n_mid = 25      # number of neuron on middle layer
n_out = 3       # number of neuron on output layer

wb_width = 0.1  # let weights and bias smaller
eta = 0.01      # Learning_rate
epoch = 1000    # epoch
interval = 100  # epoch no increase rate
batch_size = 8  # batch_size


#-- 各層的繼承來源 --
class BaseLayer:
    def __init__(self, n_upper, n):
        self.w = wb_width * np.random.randn(n_upper, n)  # 權重矩陣
        self.b = wb_width * np.random.randn(n)  # 偏值向量
        
        #######################################################
        #######################################################

    def update(self, eta):
        #######################################################
        self.w -= eta * self.grad_w

        #######################################################
        self.b -= eta * self.grad_b


# -- 中間層 --
class MiddleLayer(BaseLayer):
    def forward(self, x):  # forward_propagation
        self.x = x  # single value from input_data, save for backward_propagation
        self.u = np.dot(x, self.w) + self.b  # neuron is here
        # self.y = np.where(self.u <= 0, 0, self.u)  # ReLU
        self.y = activation_functions.relu_function(self.u)
    
    def backward(self, grad_y):  # backward_propagation
        delta = grad_y * np.where(self.u <= 0, 0, 1)  # ReLU的微分####################

        self.grad_w = np.dot(self.x.T, delta)   # the gradient for weight, size : ( * )
        self.grad_b = np.sum(delta, axis=0)     # the gradient for bias, size : ( * )
        
        self.grad_x = np.dot(delta, self.w.T)   # the gradient for input from the arrow of front neuron to here, size : ( * )

# -- 輸出層 --
class OutputLayer(BaseLayer):     
    def forward(self, x):  # forward_propagation
        self.x = x  # single value from upper layer, save for backward_propagation
        u = np.dot(x, self.w) + self.b  # neuron is here
        # self.y = np.exp(u)/np.sum(np.exp(u), axis=1, keepdims=True)  # softmax 函數
        self.y = activation_functions.softmax_function(u)

    def backward(self, t):  # backward_propagation
        delta = self.y - t  # delta mean_square_error(5-25, 5-52)##########################################
        
        self.grad_w = np.dot(self.x.T, delta)  # the gradient for weight, size : ( * )
        self.grad_b = np.sum(delta, axis=0)    # the gradient for bias, size : ( * )
        
        self.grad_x = np.dot(delta, self.w.T)  # the gradient for input from the arrow of front neuron to here, size : ( * )
        

# -- 各層的實體化 --
middle_layer_1 = MiddleLayer( n_in, n_mid)  # input neuron number,  4 to 25
middle_layer_2 = MiddleLayer(n_mid, n_mid)  # input neuron number, 25 to 25
output_layer   = OutputLayer(n_mid, n_out)  # input neuron number, 25 to  3



# -- 前向傳播 --
def forward_propagation(x):
    middle_layer_1.forward(x)
    middle_layer_2.forward(middle_layer_1.y)
    output_layer.forward(middle_layer_2.y)

# -- 反向傳播 --
def backpropagation(t):
    output_layer.backward(t)
    middle_layer_2.backward(output_layer.grad_x)
    middle_layer_1.backward(middle_layer_2.grad_x)

# -- 修正權重參數 --
def update_wb():
    middle_layer_1.update(eta)
    middle_layer_2.update(eta)
    output_layer.update(eta)

# # -- 計算誤差 --
# def get_error(t, batch_size):
#     return -np.sum(t * np.log(output_layer.y + 1e-7)) / batch_size  # 交叉熵誤差


# -- 開始訓練 --

# -- 記錄誤差用 --
train_error_x = []  # log and print at saigo
train_error_y = []  # log and print at saigo
test_error_x = []   # log and print at saigo
test_error_y = []   # log and print at saigo

# -- 記錄學習與進度--
n_batch = n_train // batch_size  # 每 1 epoch 的批次數量

for i in range(epoch):

    
    # -- 計算誤差 --  
    forward_propagation(input_train)
    # error_train = get_error()
    error_train = loss_functions.cross_entropy(output_layer.y, correct_train, n_train)
    forward_propagation(input_test)
    # error_test = get_error(correct_test, n_test)
    error_test = loss_functions.cross_entropy(output_layer.y, correct_test, n_test)
    
    # -- 記錄誤差 -- 
    test_error_x.append(i)             # log and print at saigo
    test_error_y.append(error_test)    # log and print at saigo
    train_error_x.append(i)            # log and print at saigo
    train_error_y.append(error_train)  # log and print at saigo
    
    # -- 顯示進度 -- 
    if i%interval == 0:
        print("Epoch:" + str(i) + "/" + str(epoch),
              "Error_train:" + str(error_train),
              "Error_test:" + str(error_test))

       
    
    # -- 訓練 -- 
      # randomize index
    index_random = np.arange(n_train)  # index making, 63
    np.random.shuffle(index_random)    # randomize index 

    for j in range(n_batch):
        
        # 取出小批次
        mb_index = index_random[j*batch_size : (j+1)*batch_size]####################################

        x = input_train[mb_index, :]
        t = correct_train[mb_index, :]
        
        # 前向傳播與反向傳播
        forward_propagation(x)
        backpropagation(t)
        
        # 更新權重與偏值
        update_wb() 



        
# -- 以圖表顯示誤差 -- 
plt.plot(train_error_x, train_error_y, label="Train")
plt.plot(test_error_x, test_error_y, label="Test")
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Error")

plt.show()


# -- 計算準確率 -- 
forward_propagation(input_train)
count_train = np.sum(np.argmax(output_layer.y, axis=1) == np.argmax(correct_train, axis=1))

forward_propagation(input_test)
count_test = np.sum(np.argmax(output_layer.y, axis=1) == np.argmax(correct_test, axis=1))

print("Accuracy Train:", str(count_train/n_train*100) + "%",
      "Accuracy Test:", str(count_test/n_test*100) + "%")