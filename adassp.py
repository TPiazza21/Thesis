# adaSSP

import numpy as np

# define adaSSP
def adaSSP(X,y,epsilon,delta,gamma):
    """
    # gamma is a number in [0,1], says how much budget to put in getting the hyperparameter
    """
    assert(gamma <= 1 and gamma >= 0)
    BX = 1
    BY = 1
    varrho =0.05
    [n,d] = X.shape

    epsilon_hyper = gamma * epsilon
    delta_hyper = gamma * delta
    # divide by 2 because we want all portions to add to 100%
    epsilon_necessary = (1. - gamma) * epsilon / 2.0
    delta_necessary = (1. - gamma) * delta / 2.0
    # handle the hyperparameter business (dealing with minimum eigenvalue)

    logsod_hyper = np.log(2./delta_hyper)
    eta = np.sqrt(d*logsod_hyper*np.log(2*d*d/varrho))*BX*BX/(epsilon_hyper)
    XTy = X.T.dot(y)
    # why does Wang add this identity matrix?
    XTX = (X.T).dot(X) + np.eye(d)
    w = np.linalg.eigvals(XTX)
    lambda_min_true = min(w)

    lamb_min = lambda_min_true + np.random.normal(0., 1., 1)*BX*BX*np.sqrt(logsod_hyper)/(epsilon_hyper) - logsod_hyper/(epsilon_hyper)
    lamb_min = max(lamb_min, 0.)
    lamb = max(0.0, eta - lamb_min)


    # then handle the "necessary" parts of linear regression
    logsod_necessary = np.log(2./delta_necessary)
    # for X^Ty
    normal_vec = np.random.normal(0.0,1.0,d)
    normal_vec_scalar = (np.sqrt(logsod_necessary)/(epsilon_necessary))*BX*BY
    XTy_hat = XTy + normal_vec_scalar * normal_vec

    # for X^TX
    normal_mat = np.random.normal(0.0,1.0,(d,d))

    # I should look at the coefficients of this thing! beware scalars along diagonal
    symmetric_normal_mat = 0.5 * (normal_mat + normal_mat.T)
    XTX_hat = XTX + (np.sqrt(logsod_necessary)/(epsilon_necessary))*BX*BX*symmetric_normal_mat

    # the new version of (X^TX + lambda*I)^{-1}(X^Ty)
    # you previously made a mistake where lamb was mistakenly put as lamb_min
    theta_hat = np.linalg.inv(XTX_hat + lamb * np.eye(d)).dot(XTy_hat)

    return theta_hat

# some ready made functions to try out
def adaSSP_1_3(X,y,epsilon, delta):
    return adaSSP(X=X,y=y,epsilon=epsilon,delta=delta, gamma=1.0/3.0)

def adaSSP_2_3(X,y,epsilon, delta):
    return adaSSP(X=X,y=y,epsilon=epsilon,delta=delta, gamma=2.0/3.0)

def adaSSP_1_6(X,y,epsilon, delta):
    return adaSSP(X=X,y=y,epsilon=epsilon,delta=delta, gamma=1.0/6.0)

def adaSSP_1_100(X,y,epsilon, delta):
    return adaSSP(X=X,y=y,epsilon=epsilon,delta=delta, gamma=1.0/100.0)

def adaSSP_5_6(X,y,epsilon, delta):
    return adaSSP(X=X,y=y,epsilon=epsilon,delta=delta, gamma=5.0/6.0)

