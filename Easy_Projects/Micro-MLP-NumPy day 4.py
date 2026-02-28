import numpy as np #Проект демонстрирует проход сигнала (Forward Propagation) через скрытый слой

def relu(x):
    return np.maximum(0, x)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
W_hidden = np.array([[1, -1], [-1, 1]])
b_hidden = np.array([0, 0])
W_output = np.array([1, 1])
b_output = 0
z_hidden = np.dot(X, W_hidden.T) + b_hidden
a_hidden = relu(z_hidden)


z_output = np.dot(a_hidden, W_output) + b_output

final_output = relu(z_output)

result = final_output.tolist()

print(result) 

