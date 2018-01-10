import numpy as np


#Implementing the SIGMOID function
def sigm(x, deriv=False):
    if(deriv==True):
        return (x*(1-x))

    return 1/(1+np.exp(-x))

#input data
#In this, I have divided the two input numbers into subparts. So columns 1 & 2 represent the first & second bits of number A respectively.
# Similarly, columns 3 & 4 represent first & second digits of number B respectively
# Last column is the Bias term
X = np.array([[0,0,0,0,1],
            [0,0,0,1,1],
            [0,0,1,0,1],
            [0,0,1,1,1],
            [0,1,0,0,1],
            [0,1,0,1,1],
            [0,1,1,0,1],
            [0,1,1,1,1],
            [1,0,0,0,1],
            [1,0,0,1,1],
            [1,0,1,0,1],
            [1,0,1,1,1],
            [1,1,0,0,1],
            [1,1,0,1,1],
            [1,1,1,0,1],
            [1,1,1,1,1]])


#output data for XOR
y = np.array([[0,0],
             [0,1],
             [1,0],
             [1,1],
             [0,1],
             [0,0],
             [1,1],
             [1,0],
             [1,0],
             [1,1],
             [0,0],
             [0,1],
             [1,1],
             [1,0],
             [0,1],
             [0,0]])


np.random.seed(1)   #The seed is set to return the same random numbers each time

# Initializing synapses
syn0 = 2*np.random.random((5,16)) - 1  # 5x16 matrix of weights ((4 inputs + 1 bias) x 16 nodes in the hidden layer)
syn1 = 2*np.random.random((16,2)) - 1  # 16x2 matrix of weights. (16 nodes x 2 outputs) - no bias term in the hidden layer.


#training step

for j in range(1000000):

    # Forward propagation
    lay0 = X
    lay1 = sigm(np.dot(lay0, syn0))
    lay2 = sigm(np.dot(lay1, syn1))

    # Back propagation
    lay2_error = y - lay2
    if(j % 50000) == 0:   # Only print the error every 50000 steps, to limit the amount of output.
        print("Error: " + str(np.mean(np.abs(lay2_error))))

    lay2_delta = lay2_error*sigm(lay2, deriv=True)

    lay1_error = lay2_delta.dot(syn1.T)

    lay1_delta = lay1_error * sigm(lay1,deriv=True)

    #update weights
    syn1 += lay1.T.dot(lay2_delta)
    syn0 += lay0.T.dot(lay1_delta)

print("After training, the output is:")
print(lay2)
