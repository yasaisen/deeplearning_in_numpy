
import numpy as np
import matplotlib.pyplot as plt

class loss_functions:
    def mean_square_error(y, t):
        return 1.0/2.0 * np.sum(np.square(y - t))

    def cross_entropy(y, t, batch_size=1):
        return - np.sum(t * np.log(y + 1e-7)) / batch_size

y = np.zeros(7)
t = np.zeros(7)

y = [0.1, 0.2, 0.5, 0.4, 0.4, 0.7, 0.8]
t = [0.9, 0.2, 0.7, 0.7, 0.4, 0.5, 0.8]

plt.plot(loss_functions.cross_entropy(y, t), y)
plt.title('step_function')
plt.figure()