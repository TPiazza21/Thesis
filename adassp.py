# adaSSP

import numpy as np

# define adaSSP
def adaSSP(X,y,epsilon,delta,gamma):
    """
    # gamma is a number in [0,1], says how much budget to put in getting the hyperparameter
    """
    assert(gamma <= 1 and gamma >= 0)
    BX = 1
    BY = max(y.max(), y.min(), key=abs)#1
    varrho =0.05
    [n,d] = X.shape

    epsilon_hyper = gamma * epsilon
    delta_hyper = gamma * delta
    # divide by 2 because we want all portions to add to 100%
    epsilon_necessary = (1. - gamma) * epsilon / 2.0
    delta_necessary = (1. - gamma) * delta / 2.0
    # handle the hyperparameter business (dealing with minimum eigenvalue)

    # this leading factor of 2 may not have been in the original algorithm, so be careful
    logsod_hyper = 2.0*np.log(2./delta_hyper)
    logsod_necessary = 2.0*np.log(2./delta_necessary)

    # the eta should roughly match the noise used in the release of X^TX
    eta = np.sqrt(d*logsod_necessary*np.log(2*d*d/varrho))*BX*BX/(epsilon_necessary)

    XTy = X.T.dot(y)
    # why does Wang add this identity matrix? -- for numerical stability, but this changes his algororithm
    XTX = (X.T).dot(X) + np.eye(d)
    w = np.linalg.eigvals(XTX)
    lambda_min_true = min(w)

    lamb_min = lambda_min_true + np.random.normal(0., 1., 1)*BX*BX*np.sqrt(logsod_hyper)/(epsilon_hyper) - logsod_hyper/(epsilon_hyper)
    lamb_min = max(lamb_min, 0.)
    lamb = max(0.0, eta - lamb_min)

    zero_indicator = 0
    if lamb == 0:
        zero_indicator = 1

    # then handle the "necessary" parts of linear regression
    logsod_necessary = 2.0*np.log(2./delta_necessary)
    # for X^Ty
    normal_vec = np.random.normal(0.0,1.0,d)
    normal_vec_scalar = (np.sqrt(logsod_necessary)/(epsilon_necessary))*BX*BY
    XTy_hat = XTy + normal_vec_scalar * normal_vec

    # for X^TX
    normal_mat = np.random.normal(0.0,1.0,(d,d))

    symmetric_normal_mat = np.tril(normal_mat) + np.tril(normal_mat, -1).T
    XTX_hat = XTX + (np.sqrt(logsod_necessary)/(epsilon_necessary))*BX*BX*symmetric_normal_mat

    # the new version of (X^TX + lambda*I)^{-1}(X^Ty)
    # you previously made a mistake where lamb was mistakenly put as lamb_min
    try:
        theta_hat = np.linalg.inv(XTX_hat + lamb * np.eye(d)).dot(XTy_hat)
    except:
        theta_hat = 0 # this isn't actually the right dimensions, but this has never occured
        print("Failure event!")
        assert(False)

    return theta_hat, zero_indicator

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

# this is identical to adaSSP, with the exception that we just use the constant
# for the hyperparameter, and not anything involving eigenvalue
# this is also constSSPbudget, in the paper (no longer included; constSSPfull is described later)
def constSSP(X,y,epsilon,delta, gamma):
    """
    # gamma is a number in [0,1], says how much budget to put in getting the hyperparameter
    """
    assert(gamma <= 1 and gamma >= 0)
    BX = 1
    BY = max(y.max(), y.min(), key=abs)#1
    varrho =0.05
    [n,d] = X.shape

    epsilon_hyper = gamma * epsilon
    delta_hyper = gamma * delta
    # divide by 2 because we want all portions to add to 100%
    epsilon_necessary = (1. - gamma) * epsilon / 2.0
    delta_necessary = (1. - gamma) * delta / 2.0
    # handle the hyperparameter business (dealing with minimum eigenvalue)

    logsod_hyper = 2.0*np.log(2./delta_hyper)
    logsod_necessary = 2.0*np.log(2./delta_necessary)

    # the eta should roughly match the noise used in the release of X^TX
    eta = np.sqrt(d*logsod_necessary*np.log(2*d*d/varrho))*BX*BX/(epsilon_necessary)
    XTy = X.T.dot(y)

    XTX = (X.T).dot(X) + np.eye(d)

    # this method SKIPS over this stuff with the eigenvalues, just sets it to eta
    lamb = eta

    zero_indicator = 0
    if lamb == 0:
        zero_indicator = 1

    # then handle the "necessary" parts of linear regression
    logsod_necessary = 2.0*np.log(2./delta_necessary)
    # for X^Ty
    normal_vec = np.random.normal(0.0,1.0,d)
    normal_vec_scalar = (np.sqrt(logsod_necessary)/(epsilon_necessary))*BX*BY
    XTy_hat = XTy + normal_vec_scalar * normal_vec

    # for X^TX
    normal_mat = np.random.normal(0.0,1.0,(d,d))

    symmetric_normal_mat = np.tril(normal_mat) + np.tril(normal_mat, -1).T
    XTX_hat = XTX + (np.sqrt(logsod_necessary)/(epsilon_necessary))*BX*BX*symmetric_normal_mat

    # the new version of (X^TX + lambda*I)^{-1}(X^Ty)
    try:
        theta_hat = np.linalg.inv(XTX_hat + lamb * np.eye(d)).dot(XTy_hat)
    except:
        theta_hat = 0
        print("Failure event!")
        assert(False)

    return theta_hat, zero_indicator

# same as constSSP, but uses full epsilon, delta - DP (so necessary parts have full release)
# actually, gamma plays no role here, epsilon_hyper is not used at all
def constSSPfull(X,y,epsilon,delta, gamma):
    """
    # gamma is a number in [0,1], says how much budget to put in getting the hyperparameter
    """
    assert(gamma <= 1 and gamma >= 0)
    BX = 1
    BY = max(y.max(), y.min(), key=abs)#1
    varrho =0.05
    [n,d] = X.shape

    #epsilon_hyper = gamma * epsilon
    #delta_hyper = gamma * delta

    epsilon_necessary = epsilon / 2.0
    delta_necessary = delta / 2.0
    # handle the hyperparameter business (dealing with minimum eigenvalue)

    #logsod_hyper = 2.0*np.log(2./delta_hyper)
    logsod_necessary = 2.0*np.log(2./delta_necessary)

    # the eta should roughly match the noise used in the release of X^TX
    # thus, the gamma isn't really used, because it has no bearing on anything
    eta = np.sqrt(d*logsod_necessary*np.log(2*d*d/varrho))*BX*BX/(epsilon_necessary)



    XTy = X.T.dot(y)
    XTX = (X.T).dot(X) + np.eye(d)

    # this method SKIPS over this stuff with the eigenvalues
    lamb = eta

    zero_indicator = 0
    if lamb == 0:
        zero_indicator = 1

    # then handle the "necessary" parts of linear regression
    logsod_necessary = 2.0*np.log(2./delta_necessary)

    # for X^Ty
    normal_vec = np.random.normal(0.0,1.0,d)
    normal_vec_scalar = (np.sqrt(logsod_necessary)/(epsilon_necessary))*BX*BY
    XTy_hat = XTy + normal_vec_scalar * normal_vec

    # for X^TX
    normal_mat = np.random.normal(0.0,1.0,(d,d))

    symmetric_normal_mat = np.tril(normal_mat) + np.tril(normal_mat, -1).T
    XTX_hat = XTX + (np.sqrt(logsod_necessary)/(epsilon_necessary))*BX*BX*symmetric_normal_mat

    # the new version of (X^TX + lambda*I)^{-1}(X^Ty)
    try:
        theta_hat = np.linalg.inv(XTX_hat + lamb * np.eye(d)).dot(XTy_hat)
    except:
        theta_hat = 0
        print("Failure event!")
        assert(False)

    return theta_hat, zero_indicator


