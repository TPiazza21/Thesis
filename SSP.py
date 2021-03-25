# for the SSP function
import numpy as np

# define SSP
def SSP(X,y,epsilon,delta):

    BX = 1
    BY = max(y.max(), y.min(), key=abs)
    varrho =0.05
    [n,d] = X.shape

    # divide by 2 because we want all portions to add to 100%
    epsilon_necessary = epsilon / 2.0
    delta_necessary = delta / 2.0

    XTy = X.T.dot(y)
    XTX = (X.T).dot(X) + np.eye(d) # added identity for numerical stability, but this changes OLS to ridge with hyperparameter 1

    # then handle the "necessary" parts of linear regression
    # this leading factor of 2 may be new, compared to how it is written elsewhere
    logsod_necessary = 2.0*np.log(2./delta_necessary)

    # for X^Ty
    normal_vec = np.random.normal(loc=0, scale=1, size=d)
    normal_vec_scalar = (np.sqrt(logsod_necessary)/(epsilon_necessary))*BX*BY
    XTy_hat = XTy + normal_vec_scalar * normal_vec

    # for X^TX
    normal_mat = np.random.normal(loc=0, scale=1,size=(d,d))
    symmetric_normal_mat = np.tril(normal_mat) + np.tril(normal_mat, -1).T
    XTX_hat = XTX + (np.sqrt(logsod_necessary)/(epsilon_necessary))*BX*BX*symmetric_normal_mat

    # the new version of (X^TX)^{-1}(X^Ty)
    try:
        # again, beware that XTX_hat may have an extra identity matrix added to it
        theta_hat = np.linalg.inv(XTX_hat).dot(XTy_hat)
    except:
        theta_hat = 0
        print("Failure event!")
        assert(False)

    return theta_hat
