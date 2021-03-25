# for vanilla linear regression. This is the nonprivate version
import numpy as np

def linreg(X,y,epsilon,delta):

    [n,d] = X.shape
    XTy = X.T.dot(y)
    # identity matrix added for numerical stability --> WHICH CHANGES IT TO RIDGE WITH LAMBDA=1
    XTX = (X.T).dot(X) + np.eye(d)
    theta_hat = np.linalg.inv(XTX).dot(XTy)

    return theta_hat

def ridgereg(X,y, epsilon, delta, lamb):
    [n,d] = X.shape

    XTy = X.T.dot(y)
    # maybe be careful about how you sometimes have lamb = 1 for the linear regression part, so maybe add 1 to lambda, if being consistent
    XTX = (X.T).dot(X) + lamb * np.eye(d)
    theta_hat = np.linalg.inv(XTX).dot(XTy)

    return theta_hat


