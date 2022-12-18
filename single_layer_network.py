import random

import numpy as np
import matplotlib.pyplot as plt

# def forward(x, w):
#     a = np.dot(np.transpose(w), x)
#     U = HardLim(a)
#     return U
#
# def HardLim(I):
#     if (I >= 0):
#         return 1
#     else:
#         return 0
#
#
# def main():
#
#     x = np.array([[1, 1, 1, 1],
#                   [0, 0, 1, 1],
#                   [0, 1, 0, 1]])
#     y = np.array([0, 0, 0, 1])
#     print(f"x vecor is {x}")
#     print(f"y vector is {y}")
#     W = np.random.rand(3, 1)
#     print(f"init weight matrix {W}")
#     e = 1
#     num = 4
#     while num > 0:
#         num = 0
#         for k in range(4):
#             print("------------")
#             print(f"x[:,k] is {x[:,k]}")
#             print("------------")
#             Out = forward(x[:, k], W)
#             e = y[k] - Out
#             print(f"error value is {e}")
#             if e != 0:
#                 num += 1
#             W = W + e * x[:, k:k + 1]
#
#     print(f" final weight matrix is {W}")
#     for i in range(4):
#         print(forward(x[:, i:i + 1], W))
#
#
# if __name__ == '__main__':
#     main()
def thresholding_funtion(vector):

    output_array = np.empty((vector.shape[0],1))
    print(f"output array is {output_array} with dimention {output_array.shape}")
    for num_index in range(vector.shape[0]):
        if vector[num_index] >= 0:
            output_array.put(num_index, 1)
        else:
            output_array.put(num_index, 0)
    return output_array

def learn(inputs, weights):
    print(f"wx is {np.dot(inputs, weights)}")

    return thresholding_funtion(np.dot(inputs, weights))

def train(inputs, outputs, weights, epochs):
    for epoch in range(epochs):
        output = learn(inputs, weights)
        print(f"output of learn function at epoch {epoch} is {output} with dimention {output.shape}")
        error = outputs - output
        total_error = np.sum(error)
        print(f"error at epoch  {epoch} is {error}")
        print(f"total error of all data points at epoch {epoch+1} is {total_error}")
        adjustment = np.dot(inputs.T,error)
        # x_axis = np.arange(-2,2,1)
        # plt.figure()
        # x_axis = epoch
        # print(f"x axis is {x_axis}")
        # plt.xlabel("epochs")
        # plt.ylabel("error")
        # plt.plot(x_axis, total_error,'g')
        # plt.show()
        print(f"adjustments at epoch {epoch} is {adjustment}")
        weights = weights + adjustment
        print(f"weights at epoch {epoch} is {weights}")

    return weights

def weight_init(weight_dim):
    np.random.seed(0)
    weight_matrix = np.random.rand(weight_dim,1)

    print(f"intial weight matrix is {weight_matrix}")
    # print(weight_matrix.shape)
    return weight_matrix

def main():
    training_inputs = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    # print(training_inputs)
    # print(training_inputs.shape)
    training_outputs = np.array([[0,0,0,1]]).T
    # print(training_outputs)
    # print(training_outputs.shape)
    init_weight_matrix = weight_init(training_inputs.shape[1])
    epochs = 8
    final_weights = train(training_inputs,training_outputs, init_weight_matrix,epochs)
    # test = np.array([])
    print("-----------------------------------------------------")
    print(f"final weight matrix is {final_weights}")
    print("-----------------------------------------------------")

if __name__ == '__main__':
    main()
