import numpy as np


# for evaluating performance
def test_recovery(X,y,cvo, fun_train, theta0, epsilon, delta):
    errs = []
    for [trainIdx, testIdx] in cvo:
        theta_pred = fun_train(X=X[trainIdx,:], y=y[trainIdx], epsilon=epsilon, delta=delta)
        err = np.square(np.linalg.norm(theta_pred - theta0))
        errs.append(err)
    cvErr = np.mean(errs)
    cvStd = np.std(errs)

    return errs, cvErr, cvStd

