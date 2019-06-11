'''
An XOR gate has a 0 as an output everytime the input is the same 
and 1 as an output everytime the input is not the same
0 | 0 = 0
0 | 1 = 1
1 | 0 = 1
1 | 1 = 0
A Neural Network with 
2 Neurons in the Input Layer, 
3 Neurons in 1 Hidden Layer and 
1 Neurons in the Output Layer
Trained 50k times
Sigmoid Function Activation
Learning Rate = 0.1
Zero Initialization
'''

import numpy as np
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))

InputData = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]])

TargetData = np.array([[0], [1], [1], [0]])

w1 = np.zeros((3, 2))
b1 = np.random.randn(3, 1)

w2 = np.zeros((1, 3))
b2 = np.random.randn()

iterations = 50000

lr = 0.1

costlist = []

for i in range(iterations):
    random = np.random.choice(len(InputData))

    if i % 100 == 0:
        c = 0
        for j in range(len(InputData)):
            ze1 = np.dot(w1, InputData[j].reshape(2, 1)) + b1
            ae1 = sigmoid(ze1)

            ze2 = np.dot(w2, ae1) + b2
            ae2 = sigmoid(ze2)

            c += float(np.square(ae2 - TargetData[j]))
        costlist.append(c)

    #z1 = 3,1
    z1 = np.dot(w1, InputData[random].reshape(2, 1)) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    #backprop
    dcda2 = 2 * (a2 - TargetData[random])
    da2dz2 = sigmoid_p(z2)
    dz2dw2 = a1

    dz2da1 = w2
    da1dz1 = sigmoid_p(z1)
    dz1dw1 = InputData[random].reshape(1, 2)

    w2 = (w2.T - lr * dcda2 * da2dz2 * dz2dw2).T
    b2 = b2 - lr * dcda2 * da2dz2 * 1

    w1 = w1 - np.dot((lr * dcda2 * da2dz2 * w2.T * da1dz1), dz1dw1)
    b1 = b1 - lr * dcda2 * da2dz2 * w2.T * da1dz1

print("W1: \n", w1, "\n")
print("B1: \n", b1, "\n")
print("w2: \n", w2, "\n")
print("B2: \n", b2, "\n")

for j in range(len(InputData)):
    plt.grid()
    print(InputData[j])
    ze1 = np.dot(w1, InputData[j].reshape(2, 1)) + b1
    ae1 = sigmoid(ze1)

    ze2 = np.dot(w2, ae1) + b2
    ae2 = sigmoid(ze2)

    cost = float(np.square(ae2 - TargetData[j]))
    print("Prediction: ", ae2)
    print("Cost: ", cost)
    c = 'r'
    if TargetData[j] == 0:
        c = 'b'
    plt.scatter(InputData[j][0], InputData[j][1], c=c)


for x in np.linspace(-0.25, 1.2, 20):
    for y in np.linspace(-0.25, 1.2, 20):
        x1 = np.array([[x], [y]])
        ze1 = np.dot(w1, x1) + b1
        ae1 = sigmoid(ze1)

        ze2 = np.dot(w2, ae1) + b2
        ae2 = sigmoid(ze2)
        c = 'b'
        if ae2 > 0.5:
            c = 'r'
        plt.scatter(x, y, c=c, alpha=0.2)
plt.show()

cos = plt.plot(costlist)
plt.show(cos)
