# experiments using "real" data

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt

# methods
from SSP import SSP
from linreg import linreg, ridgereg
from adassp import adaSSP, constSSP, constSSPfull, adaSSP_1_3, adaSSP_2_3, adaSSP_1_100, adaSSP_1_6, adaSSP_5_6

from test_recovery import test_recovery, test_prediction_error

import scipy.io


# set delta globally, to be used - I may also use a particular function, as needed
delta = 10 ** (-6)

def delta_func(n):
    #return 10 ** (-6) # I could hardcode it
    return n ** (-1.1)


def plots_by_gamma():
    epslist = [0.001, 0.01, 0.1,1]

    # data_name_list = ["airfoil", "autompg", "bike", "wine", "yacht"] # and "autos"? and "3droad"?
    # the ones that work are ["airfoil", "autompg", "bike", "wine", "yacht"]
    # but I only have so much space, so use these:
    data_name_list = ["bike", "wine", "yacht"]

    cross_val_splits = 32

    #gammas = [0.05 * i for i in range(1, 10)] + [0.01, 0.02, 0.03, 0.04] + [0.66, 0.95] + [0.1 ** i for i in range(3,6)] # vary by 0.05? Maybe, sure, why not. I may make this more finegrained later
    #gammas  = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    #gammas = [0.01 + 0.02 * i for i in range(50)]
    gammas = [0.05 + 0.05 * i for i in range(19)] + [0.01, 0.02, 0.03, 0.04]
    gammas.sort()
    print(f"using cross_val_splits of {cross_val_splits}, gammas are {gammas}")


    def create_budget_func(gamma):
        def temp_func(X,y,epsilon, delta):
            return adaSSP(X=X,y=y,epsilon=epsilon,delta=delta, gamma=gamma)
        return temp_func

    methodslist = [create_budget_func(gamma) for gamma in gammas]
    methodsNamelist = [f"adaSSPbudget {gamma:.4f}" for gamma in gammas]

    constSSPfull_func = lambda X,y,epsilon,delta: constSSPfull(X=X,y=y,epsilon=epsilon,delta=delta, gamma=0.3)#

    num_data = len(data_name_list)
    num_eps = len(epslist)
    num_method = len(methodslist)


    sigma = 1

    # do the SSP algorithm on its own
    #non_private_results = np.zeros((num_eps, num_data))
    SSP_results = np.zeros((num_eps, num_data))
    SSP_std = np.zeros((num_eps, num_data))

    results_err = np.zeros((num_method,num_eps,num_data))
    results_std = np.zeros((num_method,num_eps,num_data))

    # for the const values (the constSSP)
    #const_results_err = np.zeros((num_method,num_eps,num_data))
    #const_results_std = np.zeros((num_method,num_eps,num_data))

    const_full_results_err = np.zeros((num_eps,num_data))
    const_full_results_std = np.zeros((num_eps,num_data))

    zero_error_counts = np.zeros((num_method,num_eps,num_data))

    d_map = {}
    n_map = {}

    for i in range(num_data):
        # extract and parse data
        data_name = data_name_list[i]
        print("data is " + data_name)
        all_data = scipy.io.loadmat('data/data_uci/' + data_name + '/' + data_name + '.mat')['data']

        n = all_data.shape[0]
        d = all_data.shape[1] - 1 # minus 1, because of the y part

        d_map[i] = d
        n_map[i] = n

        X = all_data[:,:d]
        y = all_data[:,-1]

        X = X - X.mean()
        y = y - y.mean()
        y = y / y.std()

        X_sqrt_sum = np.sqrt(np.sum(np.square(X), axis=1))
        X = X / X_sqrt_sum[:,None]

        all_indices = np.arange(n)
        cvo = [splits for splits in ShuffleSplit(n_splits=cross_val_splits, test_size=0.1).split(all_indices)]

        for j in range(num_eps):
        #% set parameters of the algorithm
            eps = epslist[j]
            print(f"eps is {eps:.4f}")

            # may as well also track SSP
            t = time.time()
            errs, cvErr,cvStd, errorDict = test_prediction_error(X, y, cvo, SSP, eps, delta, d)
            assert(not np.isnan(cvErr))
            t_run = time.time() - t
            print(f"method = SSP, Time run was {t_run:.4f}s")
            SSP_results[j, i] = cvErr
            SSP_std[j, i] = cvStd

            # constSSPfull
            errs, cvErr,cvStd, errorDict = test_prediction_error(X, y, cvo, constSSPfull_func, eps, delta, d)
            assert(not np.isnan(cvErr).any())
            const_full_results_err[j,i] = cvErr
            const_full_results_std[j,i] = cvStd

            # then for the rest of them
            for k in range(num_method):
                fun = methodslist[k]
                t = time.time()
                errs, cvErr,cvStd, errorDict = test_prediction_error(X, y, cvo, fun, eps, delta, d)
                assert(not np.isnan(cvErr).any())
                t_run = time.time() - t
                print(f"method = {methodsNamelist[k]}, Time run was {t_run:.4f}s")
                results_err[k,j,i] = cvErr
                results_std[k,j,i] = cvStd
                # the "lambda zero proportion" from the thesis
                zero_error_counts[k,j,i] = errorDict["zero_counter"]

    # plotting prediction error
    fig, axs = plt.subplots(2, 2, figsize=(16,14), sharex=True, sharey=True)
    fig.suptitle('Prediction Error (MSE), UCI Data', fontsize=20)
    axs_raveled = list(np.ravel(axs))
    for j in range(num_eps):
        eps = epslist[j]
        ax = axs_raveled[j]

        # instead of a different curve for each method, do a different curve for each n
        for i in range(num_data):
            # plot gammas on the x-axis
            adaSSPbudget_yerr = np.concatenate(    (np.zeros((num_method, 1)), results_std[:,j,i].reshape((num_method,1))),    axis=1).transpose()
            ax.errorbar(x=gammas, y=results_err[:,j,i], yerr=adaSSPbudget_yerr, label=f"adaSSPbudget, data={data_name_list[i]}, n={n_map[i]}, d={d_map[i]}")

            #constant set to 3 because of SSP, adaSSPbudget, constSSPfull
            temp_color = ax.get_lines()[3*i].get_color()

            constSSPfull_vals = [const_full_results_err[j,i]] * len(gammas)
            constSSPfull_stds = np.array([const_full_results_std[j,i]] * len(gammas))
            constSSPfull_yerr = np.concatenate(    (np.zeros((num_method, 1)), constSSPfull_stds.reshape((num_method,1))),    axis=1).transpose()
            ax.errorbar(x=gammas, y=constSSPfull_vals, yerr=constSSPfull_yerr, label=f"constSSPfull, data={data_name_list[i]}, n={n_map[i]}, d={d_map[i]}", color=temp_color, linestyle="dotted")

            ssp_vals = [SSP_results[j,i]] * len(gammas)
            ssp_stds = np.array([SSP_std[j,i]] * len(gammas))
            ax.errorbar(x=gammas, y=ssp_vals, yerr=np.concatenate(    (np.zeros((num_method, 1)), ssp_stds.reshape((num_method,1))),    axis=1).transpose(), label=f"SSP, data = {data_name_list[i]}", color=temp_color, linestyle="dashed")

        ax.plot([1.0/3.0, 1.0/3.0], [0, 10** 5], label="adaSSP gamma value")
        ax.set_xlabel("Gamma")
        ax.set_ylabel('Prediction Error (MSE)')
        ax.set_title(f"epsilon = {eps:.3f}")
        ax.set_yscale('log')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"plots/pred_err_by_gamma_all_eps_data.png")
    plt.close()

    # for plotting errors
    for j in range(num_eps):
        fig, ax = plt.subplots(1, figsize=(9,6)) # just one, for now
        eps = epslist[j]

        # instead of a different curve for each method, do a different curve for each n
        for i in range(num_data):
            # plot gammas on the x-axis
            error_props = [val / cross_val_splits for val in list(zero_error_counts[:,j,i])]
            ax.plot(gammas,error_props, label=f"data={data_name_list[i]}, n={n_map[i]}, d={d_map[i]}", alpha=0.7)

        ax.set_xlabel("Gamma")
        ax.set_ylabel('Lambda Zero Proportion (for adaSSPbudget)')
        ax.set_title(f"Lambda Zero Proportion For epsilon = {eps:.3f}, UCI Data")
        ax.legend()
        plt.savefig(f"plots/zero_count_by_gamma_eps_{eps:.4f}_data.png")

"""
# this checks non private ridge regression
def non_private_checks():

    cross_val_splits = 6 # maybe crank up for real results...

    data_name_list = ["airfoil", "autompg", "bike", "wine", "yacht"]

    lambs = [10 ** i for i in range(-5,5)]
    lambs.sort()

    #vanilla_index = methodsNamelist.index('Non-Private')
    #if vanilla_index < 0:
        #print("You need the nonprivate version to compute relative efficiency!")
        #assert(False)

    def create_ridge_func(lamb):
        def temp_func(X,y,epsilon, delta):
            return ridgereg(X=X,y=y,epsilon=epsilon,delta=delta, lamb=lamb)
        return temp_func

    methodslist = [create_ridge_func(lamb) for lamb in lambs]
    methodsNamelist = [f"Ridge Regression {lamb:.4f}" for lamb in lambs]


    num_data = len(data_name_list)
    num_method = len(methodslist)


    sigma = 1


    linreg_results = np.zeros((num_data))
    linreg_std = np.zeros((num_data))

    ridge_results = np.zeros((num_method, num_data))
    ridge_std = np.zeros((num_method, num_data))

    zero_error_counts = np.zeros((num_method, num_data))

    d_map = {}
    n_map = {}


    for i in range(num_data):
        data_name = data_name_list[i]
        print("data is " + data_name)
        all_data = scipy.io.loadmat('data/data_uci/' + data_name + '/' + data_name + '.mat')['data']

        n = all_data.shape[0]
        d = all_data.shape[1] - 1 # minus 1, because of the y part

        d_map[i] = d
        n_map[i] = n

        X = all_data[:,:d]
        y = all_data[:,-1]

        X = X - X.mean() # do I get mean for free?
        y = y - y.mean()
        y = y / y.std()

        X_sqrt_sum = np.sqrt(np.sum(np.square(X), axis=1))
        X = X / X_sqrt_sum[:,None]

        #cvo = cvpartition(length(y),'KFold',10);
        all_indices = np.arange(n)
        cvo = [splits for splits in ShuffleSplit(n_splits=cross_val_splits, test_size=0.1).split(all_indices)] #, random_state=0) # I could put random state here, if I wanted

        #delta = 1 / (n ** (1.1)) # not so sure what this is about
        delta = delta_func(n)
        eps = 1.0

        # have to do the linreg version
        t = time.time()
        errs, cvErr,cvStd, errorDict = test_prediction_error(X, y, cvo, linreg, eps, delta, d)
        assert(not np.isnan(cvErr))
        t_run = time.time() - t
        print(f"method = linreg, Time run was {t_run:.4f}s")
        linreg_results[i] = cvErr
        linreg_std[i] = cvStd

        # then for the rest of them
        for k in range(num_method):
            fun = methodslist[k]
            t = time.time()
            errs, cvErr,cvStd, errorDict = test_prediction_error(X, y, cvo, fun, eps, delta, d)
            assert(not np.isnan(cvErr))
            #t_run=toc; # this is for timing, which I am not keeping track of
            t_run = time.time() - t
            print(f"method = {methodsNamelist[k]}, Time run was {t_run:.4f}s")
            #fprintf('%s at eps = %f: Test err = %.2f, std = %.2f, runtime = %.2f s.\n', methodsNamelist{k}, opts.eps, cvErr,cvStd,t_run)
            ridge_results[k,i] = cvErr
            ridge_std[k,i] = cvStd

    fig, ax = plt.subplots(1, figsize=(9,6)) # just one, for now


    # instead of a different curve for each method, do a different curve for each n
    #for k in range(num_method):
    for i in range(num_data):

        # plot gammas on the x-axis
        ax.errorbar(x=lambs, y=ridge_results[:,i], yerr=np.concatenate(    (np.zeros((num_method, 1)), ridge_std[:,i].reshape((num_method,1))),    axis=1).transpose(), label=f"data = {data_name_list[i]}, n={n_map[i]}, d={d_map[i]}")

        temp_color = ax.get_lines()[2*i].get_color()#ax.get_color()

        #ssp_vals = [SSP_rel_eff[j,i]] * len(gammas)
        #ssp_stds = np.array([SSP_rel_eff_std[j,i]] * len(gammas))
        linreg_vals_display = [linreg_results[i]] * len(lambs)
        linreg_stds_display = np.array([linreg_std[i]] * len(lambs))
        #ax.plot([0,1], [SSP_rel_eff[j,i], SSP_rel_eff[j,i]], label=f"SSP, n = {nlist[i]}", color=temp_color, linestyle="dashed")

        ax.errorbar(x=lambs, y=linreg_vals_display, yerr=np.concatenate(    (np.zeros((num_method, 1)), linreg_stds_display.reshape((num_method,1))),    axis=1).transpose(), label=f"linreg, data = {data_name_list[i]}, n={n_map[i]}, d={d_map[i]}", color=temp_color, linestyle="dashed")

    #ax.set_xlabel("n values (size of data set)")
    ax.set_xlabel("Ridge Regression Lambda Parameter")
    ax.set_ylabel('Prediction Error (MSE)')
    ax.set_title(f"Prediction Error, UCI Data, Nonprivate Methods (Ridge and Linear Regression)")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    plt.savefig(f"plots/pred_err_nonprivate_data.png")
"""

if __name__ == "__main__":
    plots_by_gamma()
    #non_private_checks()

