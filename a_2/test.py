import numpy as np

# Paste your sigmoid function here
def nnObjFunction(params, *args):
    '''
    % nnObjFunction computes the value of objective function (cross-entropy
    % with regularization) given the weights and the training data and lambda
    % - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices W1 (weights of connections from
    %     input layer to hidden layer) and W2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not including the bias node)
    % n_hidden: number of node in hidden layer (not including the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % train_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % train_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector (not a matrix) of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.
    '''
    n_input, n_hidden, n_class, train_data, train_label, lambdaval = args
    # First reshape 'params' vector into 2 matrices of weights W1 and W2
    W1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))  # Input --> hidden
    W2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))  # Hidden --> output
    obj_val = 0
    # Feed Forwards Algo:
    # Calculating hidden layer nodes (1+2)
    i_data_bias = np.ones((len(train_data), 1))
    train_data = np.concatenate((train_data, i_data_bias), 1)  # adding bias term to x
    z = sigmoid(np.dot(train_data, np.transpose(W1)))
    # Calculating output layer nodes (3+4)
    h_data_bias = np.ones((len(z), 1))
    z = np.concatenate((z, h_data_bias), 1)  # adding bias term to z
    o = sigmoid(np.dot(z, np.transpose(W2)))
    # 1-to-K encoding:
    one_to_k = np.eye(len(o), len(o[0]))[train_label]
    # Computing Error Function (5 + 15):
    obj_val = -1.0 * (np.sum(one_to_k * np.log(o) + (1 - one_to_k) * np.log(1 - o)) / len(train_data))
    obj_val_reg = obj_val + (lambdaval * (np.sum(W1**2) + np.sum(W2**2))) / (2 * len(train_data))
    # TODO Backpropogation:
    delta = o - one_to_k  # o-y
    hidden = (1 - z) * z * np.dot(delta, W2)
    grad_w1 = np.dot(np.transpose(hidden), train_data)
    deriv_W1 = (np.delete(grad_w1, len(grad_w1) - 1, 0) + lambdaval * W1) / len(train_data)
    deriv_W2 = (np.dot(np.transpose(delta), z) + lambdaval * W2) / len(train_data)
    obj_grad = np.concatenate((deriv_W1.flatten(), deriv_W2.flatten()), 0)
    return obj_val_reg, obj_grad


# Paste your nnObjFunction here
def sigmoid(z):
    '''
    Notice that z can be a scalar, a vector or a matrix
    return the sigmoid of input z (same dimensions as z)
    '''
    s = 1.0/(1.0+np.exp(-1.0*z))
    return s


n_input = 5
n_hidden = 3
n_class = 2
training_data = np.array([np.linspace(0,1,num=5),np.linspace(1,0,num=5)])
training_label = np.array([0,1])
lambdaval = 1
params = np.linspace(-5,5, num=26)
args = (n_input, n_hidden, n_class, training_data, training_label, lambdaval)
objval,objgrad = nnObjFunction(params, *args)
print("Objective value:")
print(objval)
print("Gradient values: ")
print(objgrad)