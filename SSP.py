# for the SSP function
import numpy as np

# define SSP
def SSP(X,y,epsilon,delta):

    BX = 1
    BY = 1
    varrho =0.05
    [n,d] = X.shape

    # divide by 2 because we want all portions to add to 100%
    epsilon_necessary = epsilon / 2.0
    delta_necessary = delta / 2.0
    # handle the hyperparameter business (dealing with minimum eigenvalue)


    XTy = X.T.dot(y)
    # why this added identity matrix?
    XTX = (X.T).dot(X) + np.eye(d)

    # then handle the "necessary" parts of linear regression
    logsod_necessary = np.log(2./delta_necessary)
    # for X^Ty
    normal_vec = np.random.normal(loc=0, scale=1, size=d)
    normal_vec_scalar = (np.sqrt(logsod_necessary)/(epsilon_necessary))*BX*BY
    XTy_hat = XTy + normal_vec_scalar * normal_vec

    # for X^TX
    normal_mat = np.random.normal(loc=0, scale=1,size=(d,d))
    #symmetric_normal_mat = 0.5 * (normal_mat + normal_mat.T)
    symmetric_normal_mat = np.tril(normal_mat) + np.tril(normal_mat, -1).T
    XTX_hat = XTX + (np.sqrt(logsod_necessary)/(epsilon_necessary))*BX*BX*symmetric_normal_mat

    # the new version of (X^TX)^{-1}(X^Ty)
    theta_hat = np.linalg.inv(XTX_hat).dot(XTy_hat)

    return theta_hat
