__author__ = 'Antares'

import numpy as np
from minimize import minimize


def fold(dna, shape_theta, shape_x):
    """
    convert 1-dimension dna into theta and x

    @param dna:
    @param shape_theta:
    @param shape_x:
    @return:
    """
    # check length
    assert dna.size == np.product(shape_theta) + np.product(shape_x)
    theta = dna[0: np.product(shape_theta)].reshape(shape_theta)
    x = dna[np.product(shape_theta):].reshape(shape_x)

    return theta, x


def unfold(theta, x):
    """
    convert theta and x into a 1-dimension dna
    @param theta:
    @param x:
    @return:
    """
    return np.concatenate((theta.flatten(), x.flatten()))


def cost_function(dna, shape_theta, shape_x, y, r, reg_lambda):
    theta, x = fold(dna, shape_theta, shape_x)

    # cost
    d = (x.dot(theta.T)-y) * r
    cost = (1/2)*np.sum(d**2) + (reg_lambda/2)*np.sum(theta**2) + (reg_lambda/2)*np.sum(x**2)

    # gradient
    theta_gradient = d.T.dot(x) + reg_lambda*theta
    x_gradient = d.dot(theta) + reg_lambda*x
    grad = np.concatenate((theta_gradient.flatten(), x_gradient.flatten()))

    return cost, grad


def learn(shape_theta, shape_x, y, r, reg_lambda, n_iter):
    num_movies = y.shape[0]
    num_users = y.shape[1]

    # Normalize Ratings
    y_sum = y.sum(axis=1)
    r_sum = r.sum(axis=1)
    r_sum += (r_sum<=0)
    y_mean = (y_sum/r_sum).reshape((-1, 1))
    y = y - y_mean.dot(np.ones((1, num_users)))

    param_0 = np.random.randn(np.product(shape_theta) + np.product(shape_x))

    # optimize
    opt, cost, i = minimize(lambda dna: cost_function(dna, shape_theta, shape_x, y, r, reg_lambda),
                            param_0,
                            n_iter)

    theta, x = fold(opt, shape_theta, shape_x)
    reg_cost = (reg_lambda/2)*np.sum(theta**2) + (reg_lambda/2)*np.sum(x**2)

    return theta, x, y_mean, cost, reg_cost


def main():
    pass


if __name__ == "__main__":
    main()