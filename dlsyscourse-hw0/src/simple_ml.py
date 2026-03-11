import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    def get_np_dtype_str(datatype_encoded: int):
        match datatype_encoded:
            case 0x08:
                dtype = "B"
            case 0x09:
                dtype = "b"
            case 0x0B:
                dtype = "h"
            case 0x0C:
                dtype = "i"
            case 0x0D:
                dtype = "f"
            case 0x0E:
                dtype = "d"
        return f">{dtype}"

    with gzip.open(image_filename, "rb") as f:
        magic, datatype_encoded, num_dims = struct.unpack(">HBB", f.read(4))
        assert magic == 0
        np_dtype_str = get_np_dtype_str(datatype_encoded)
        dims = []
        for x in range(num_dims):
            if x > 1:
                dims[1] *= struct.unpack(">I", f.read(4))[0]
            else:
                dims.append(struct.unpack(">I", f.read(4))[0])
        image_np_array = np.frombuffer(f.read(), dtype=np_dtype_str)
        min_image_val = np.min(image_np_array)
        max_image_val = np.max(image_np_array)
        image_np_array = ((image_np_array - min_image_val) / (max_image_val - min_image_val)).astype("float32")
        image_np_array = image_np_array.reshape(*dims)
    with gzip.open(label_filename, "rb") as f:
        magic, datatype_encoded, num_dims = struct.unpack(">HBB", f.read(4))
        assert magic == 0
        np_dtype_str = get_np_dtype_str(datatype_encoded)
        dims = []
        for x in range(num_dims):
            if x > 1:
                dims[1] *= struct.unpack(">I", f.read(4))[0]
            else:
                dims.append(struct.unpack(">I", f.read(4))[0])
        label_np_array = np.frombuffer(f.read(), dtype=np_dtype_str).reshape(*dims)
    return (image_np_array, label_np_array)
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    # Formula: loss = -hy(x) + log(sum(exp(hj(x))))
    return np.average(np.log(np.sum(np.exp(Z), axis=1)) - Z[np.arange(y.shape[0]), y]) # indexes into Z using y
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    k = theta.shape[1]
    def softmax(v):
        v_max = np.max(v, axis=1)
        v_max_broadcasted = np.column_stack([v_max] * 10)
        exp_v = np.exp(v - v_max_broadcasted)
        return exp_v / np.sum(exp_v, axis=1)[:, None]
    one_hot_y = np.zeros((y.size, k))
    one_hot_y[np.arange(y.size), y] = 1
    num_batches = X.shape[0] // batch
    # Initially I tried performing a single batch update by reshaping the dataset into
    # (num_batches, batch, n), but that did not work (jumped too far).
    for x in range(num_batches):
        minibatch_x = X[x * batch: (x+1) * batch].reshape(batch, X.shape[1])
        minibatch_y = one_hot_y[x * batch: (x+1) * batch]
        XT = minibatch_x.transpose()
        Z = softmax(np.matmul(minibatch_x, theta))
        gradient = np.matmul(XT, (Z - minibatch_y).reshape(batch, theta.shape[1]))
        theta -= lr * gradient / batch
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    k = W2.shape[1]
    def softmax(v):
        v_max = np.max(v, axis=1)
        v_max_broadcasted = np.column_stack([v_max] * k)
        exp_v = np.exp(v - v_max_broadcasted)
        return exp_v / np.sum(exp_v, axis=1)[:, None]
    one_hot_y = np.zeros((y.size, k))
    one_hot_y[np.arange(y.size), y] = 1
    num_batches = X.shape[0] // batch
    for x in range(num_batches):
        minibatch_x = X[x * batch: (x+1) * batch].reshape(batch, X.shape[1])
        minibatch_y = one_hot_y[x * batch: (x+1) * batch]
        layers = [W1, W2]
        # We want to cache both Zi and the next computed preactivation ZiWi
        Z = [(minibatch_x, np.matmul(minibatch_x, W1))]
        """
        Full neural network implementation w/ backprop
        """
        # Forward pass
        for ind, layer in enumerate(layers):
            prev_z, pre_activation_z = Z[ind]
            # Replace last layer ReLU with softmax
            if ind < len(layers) - 1:
                next_z = np.maximum(0, pre_activation_z)
            else:
                next_z = softmax(pre_activation_z) - minibatch_y
            if ind < len(layers) - 1:
                next_pre_activation_z = np.matmul(next_z, layers[ind + 1])
            else:
                next_pre_activation_z = None
            Z.append((next_z, next_pre_activation_z))
        # Backwards pass
        G = [None] * len(Z)
        G[-1] = Z[-1][0]
        for ind, layer in enumerate(layers[::-1]):
            curr_ind = len(Z) - ind - 2
            curr_z, pre_activation_z = Z[curr_ind]
            if ind == 0:
                temp_g = G[curr_ind + 1]
            else:
                temp_g = G[curr_ind + 1] * np.where(pre_activation_z > 0, 1.0, 0.0)
            G[curr_ind] = np.matmul(temp_g, layer.transpose())
            layer -= lr * np.matmul(curr_z.transpose(), temp_g) / batch

        """
        Hardcoded variant for verification

        xw1 = np.matmul(minibatch_x, W1)
        z2 = np.maximum(0, xw1)
        s_minus_i = softmax(np.matmul(z2, W2)) - minibatch_y
        grad_w2 = np.matmul(z2.transpose(), s_minus_i)
        grad_w1 = np.matmul(minibatch_x.transpose(), np.matmul(s_minus_i, W2.transpose()) * np.where(xw1 > 0, 1.0, 0.0))
        W2 -= lr * grad_w2 / batch
        W1 -= lr * grad_w1 / batch
        """
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1, cpp=False)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
