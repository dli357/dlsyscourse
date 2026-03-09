#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

float dot_product(float* x, float* y, size_t n) {
    float acc = 0;
    for (int i = 0; i < n; i++) {
        acc += x[i] * y[i];
    }
    return acc;
}


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    // Randomly shuffle samples into batches
    size_t* shuffled_indices = new size_t[m];
    for (size_t i = 0; i < m; i++) {
        shuffled_indices[i] = i;
    }
    srand (time(NULL));
    for (size_t i = 0; i < m; i++) {
        size_t rand_ind = rand() % (m - i);
        size_t tmp = shuffled_indices[rand_ind];
        shuffled_indices[rand_ind] = shuffled_indices[i];
        shuffled_indices[i] = tmp;
    }
    // Loop through each batch.
    size_t num_batches = m / batch;
    for (size_t i = 0; i < num_batches; i++) {
        size_t batch_base_ind = i * batch * n;
        // Loop through each example within a single batch.
        std::vector<float> gradients;
        gradients.reserve(ki);
        for (size_t j = 0; j < batch; j++) {
            size_t item_ind = batch_base_ind + j;
            std::vector<float> intermediates;
            intermediates.reserve(ki);
            float total_softmax = 0;
            // Compute individual softmax for each class K
            for (size_t ki = 0; ki < k; ki++) {
                float softmax_i = std::exp(dot_product(x + item_ind, theta + ki, n));
                total_softmax += softmax_i;
                intermediates.push_back(softmax_i);
            }
            for (size_t ki = 0; ki < k; ki++) {
                intermediates[ki] /= total_softmax;
            }
            // Compute cost based on resulting vs expected
            float total_cost = 0;
            for (size_t ki = 0; ki < k; ki++) {
                float cost_base = 0;
                if (y[item_ind] == ki) {
                    cost_base = 1;
                }
                gradients[ki] += cost_base - intermediates[ki];
            }
        }
        // Average the gradients for the minibatch.
        for (size_t ki = 0; ki < k; ki++) {
            theta[ki] -= gradients[ki] / batch * lr;
        }
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
