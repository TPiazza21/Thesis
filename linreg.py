# for vanilla linear regression. This is the nonprivate version
import numpy as np

def linreg(X,y,epsilon,delta):

    [n,d] = X.shape

    XTy = X.T.dot(y)
    # identity matrix added for numerical stability
    XTX = (X.T).dot(X) + np.eye(d)
    theta_hat = np.linalg.inv(XTX).dot(XTy)

    return theta_hat
