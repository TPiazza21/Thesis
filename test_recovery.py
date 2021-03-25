import numpy as np
from sklearn.metrics import mean_squared_error

# for evaluating performance
# this will compute relative efficiency
def test_recovery(X,y,cvo, fun_train, theta0, epsilon, delta):
    errs = []
    zero_counter = 0
    for [trainIdx, testIdx] in cvo:
        fun_results = fun_train(X=X[trainIdx,:], y=y[trainIdx], epsilon=epsilon, delta=delta)
        if len(list(fun_results)) == len(list(theta0)):
            theta_pred = fun_results
            zero_indicator = 0
        else:
            # keep track of "lambda zero proportions" counts
            theta_pred, zero_indicator = fun_results

        err = np.square(np.linalg.norm(theta_pred - theta0))
        errs.append(err)
        zero_counter += zero_indicator

    cvErr = np.mean(errs)
    cvStd = np.std(errs)

    error_dict = {}
    # "lambda zero proportion" counts...
    error_dict["zero_counter"] = zero_counter

    return errs, cvErr, cvStd, error_dict

# pass in d, to help with parsing the arguments of the predicted values
# this will compute mean prediction error
def test_prediction_error(X,y,cvo, fun_train, epsilon, delta, d):
    errs = []
    zero_counter = 0
    for [trainIdx, testIdx] in cvo:
        fun_results = fun_train(X=X[trainIdx,:], y=y[trainIdx], epsilon=epsilon, delta=delta)
        if len(list(fun_results)) == d:
            theta_pred = fun_results
            zero_indicator = 0
        else:
            theta_pred, zero_indicator = fun_results

        # ok, now need to see what it does on test idx
        test_pred = X[testIdx,:].dot(theta_pred)
        test_true = y[testIdx]
        err = mean_squared_error(test_true, test_pred)/np.var(y) # MSE of 1 is predicting mean, given this division
        errs.append(err)
        zero_counter += zero_indicator
    cvErr = np.mean(errs)
    cvStd = np.std(errs)

    error_dict = {}
    error_dict["zero_counter"] = zero_counter

    return errs, cvErr, cvStd, error_dict


