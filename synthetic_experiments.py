# Tyler Piazza, 11/28/2020
# this file is meant to generate synthetic data and test various DP ridge regression techniques

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

delta = 10 ** (-6) # set delta globally (used to be 1/(n^1.1), which had some issues as n grew)

def delta_func(n):
    #return 10 ** (-6)
    return n ** (-1.1)




# I may want to set a seed early on, so I have reproducible results

"""
def plots_by_n():
    d = 10
    epslist = [0.1,1]
    #nlist= 20*2.^[1:2:18];
    nlist = [20 * (2 ** i) for i in range(1,16,2)] # smaller n go faster

    cross_val_splits = 6


    #methodslist = [@suffstats_perturb,@adassp, @budget_adassp_1_6, @budget_adassp_1_3,...
        #@budget_adassp_1_2, @budget_adassp_2_3, @budget_adassp_5_6, @budget_adassp_1_100]
    methodsNamelist = ['Non-Private', 'SSP', 'Budget AdaSSP 1/3 (original)', 'Budget AdaSSP 2/3',
    'Budget AdaSSP 1/100', 'Budget AdaSSP 1/6', 'Budget AdaSSP 5/6'] #, 'Budget AdaSSP 1/6', 'Budget AdaSSP 1/3', 'Budget AdaSSP 1/2',
    #'Budget AdaSSP 2/3', 'Budget AdaSSP 5/6', 'Budget AdaSSP 1/100']
    methodslist = [linreg, SSP, adaSSP_1_3, adaSSP_2_3, adaSSP_1_100, adaSSP_1_6, adaSSP_5_6]

    vanilla_index = methodsNamelist.index('Non-Private')
    if vanilla_index < 0:
        print("You need the nonprivate version to compute relative efficiency!")
        assert(False)



    num_n = len(nlist)
    num_eps= len(epslist);
    num_method = len(methodsNamelist);

    sigma = 1

    #theta = randn(d,1);
    theta = np.random.normal(loc=0, scale=1, size=d)
    theta_norm = np.linalg.norm(theta)
    theta = theta / (theta_norm * np.sqrt(2.0))

    results_err = np.zeros((num_method,num_eps,num_n))
    results_std = np.zeros((num_method,num_eps,num_n))


    zero_error_counts = np.zeros((num_method,num_eps,num_n))

    #for i = 1:num_n
    for i in range(num_n):
        n = nlist[i]
        print(f"n is {n}")

        #X = randn(n,d);
        X = np.random.normal(loc=0, scale=1, size=(n,d))

        X_sqrt_sum = np.sqrt(np.sum(np.square(X), axis=1))
        print("X_sqrt_sum is this")
        print(X_sqrt_sum)

        #X = bsxfun(@rdivide, X,sqrt(sum(X.^2,2)));
        #https://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
        X = X / X_sqrt_sum[:,None]

        X_sqrt_sum = np.sqrt(np.sum(np.square(X), axis=1))

        print("new X_sqrt_sum is ")
        print(X_sqrt_sum)
        #y = X*theta + 0.1*sigma*(rand(n,1)-0.5);
        y = X.dot(theta) + 0.1 * sigma * (np.random.normal(loc=-0.5, scale=1, size=n))
        #y = X.dot(theta) + 0.1 * sigma * (np.random.normal(loc=0.0, scale=1, size=n))
        # shouldn't I standardize this, so that y has
        #y = y - y.mean()
        #y = y / y.std()

        #cvo = cvpartition(length(y),'KFold',10);
        all_indices = np.arange(n)
        cvo = [splits for splits in ShuffleSplit(n_splits=cross_val_splits, test_size=0.1).split(all_indices)] #, random_state=0) # I could put random state here, if I wanted


        for j in range(num_eps):
        #% set parameters of the algorithm
            eps = epslist[j]
            print(f"eps is {eps:.4f}")
            #opts.eps = epslist(j);
            #opts.delta =1/n^(1.1);
            #delta = 1 / (n ** (1.1)) # not so sure what this is about

            for k in range(num_method):
                fun = methodslist[k]
                t = time.time()
                errs, cvErr,cvStd, errorDict = test_recovery(X, y, cvo, fun, theta, eps, delta)
                assert(not np.isnan(cvErr))
                #t_run=toc; # this is for timing, which I am not keeping track of
                t_run = time.time() - t
                print(f"method = {methodsNamelist[k]}, Time run was {t_run:.4f}s")
                #fprintf('%s at eps = %f: Test err = %.2f, std = %.2f, runtime = %.2f s.\n', methodsNamelist{k}, opts.eps, cvErr,cvStd,t_run)
                results_err[k,j,i] = cvErr
                results_std[k,j,i] = cvStd

                zero_error_counts[k,j,i] = errorDict["zero_counter"]


    # maybe save('exp_gaussian.mat','results_err','results_std')

    #RelEfficiency =  bsxfun(@rdivide,results_err,results_err(2,:,:));
    # divide everything by the error for the vanilla linear regression

    rel_efficiency = np.zeros((num_method, num_eps, num_n))
    rel_efficiency_std = np.zeros((num_method, num_eps, num_n))
    vanilla_values = results_err[vanilla_index,:,:]


    for i in range(num_n):
        for j in range(num_eps):
            for k in range(num_method):
                rel_efficiency[k,j,i] = results_err[k,j,i] / vanilla_values[j,i]
                rel_efficiency_std[k,j,i] = results_std[k,j,i] / vanilla_values[j,i]


    # then for the plotting
    # right now, it's one plot per epsilon
    # the relative efficiency plots

    for j in range(num_eps):
        fig, ax = plt.subplots(1) # just one, for now
        eps = epslist[j]

        for k in range(num_method):

            #print("yerr is ")
            #print(np.concatenate(    (rel_efficiency_std[k,j,:].reshape((num_n,1)), np.zeros((num_n, 1))),    axis=1).reshape((2, num_n)))

            ax.errorbar(x=nlist, y=rel_efficiency[k,j,:], yerr=np.concatenate(    (np.zeros((num_n, 1)), rel_efficiency_std[k,j,:].reshape((num_n,1))),    axis=1).transpose(), label=methodsNamelist[k])

        ax.set_xlabel("n values (size of data set)")
        ax.set_ylabel('Relative Efficiency')
        ax.set_title(f"Relative Efficiency For epsilon = {eps:.3f}, Synthetic Data")
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()
        plt.savefig(f"plots/rel_eff_eps_{eps:.4f}_synthetic.png")
"""

def plots_by_gamma(correlated_data=True):
    if correlated_data:
        data_description = "Correlated"
    else:
        data_descriotion = "Clean"



    d = 10
    epslist = [0.001, 0.01, 0.1,1]
    #epslist = [0.01, 0.1]
    #nlist= 20*2.^[1:2:18];
    #nlist = [20 * (2 ** i) for i in range(1,16,2)] # smaller n go faster

    #nlist = [20 * (2 ** i) for i in [6, 10, 14, 18]] # maybe have 18... # start small
    nlist = [20 * (2 ** i) for i in [6, 10, 14]]
    #cross_val_splits = 10 # maybe crank up for real results...
    #cross_val_splits=32
    cross_val_splits=6


    #methodslist = [@suffstats_perturb,@adassp, @budget_adassp_1_6, @budget_adassp_1_3,...
        #@budget_adassp_1_2, @budget_adassp_2_3, @budget_adassp_5_6, @budget_adassp_1_100]
    #methodsNamelist = ['Non-Private', 'SSP', 'Budget AdaSSP 1/3 (original)', 'Budget AdaSSP 2/3',
    #'Budget AdaSSP 1/100', 'Budget AdaSSP 1/6', 'Budget AdaSSP 5/6'] #, 'Budget AdaSSP 1/6', 'Budget AdaSSP 1/3', 'Budget AdaSSP 1/2',
    #'Budget AdaSSP 2/3', 'Budget AdaSSP 5/6', 'Budget AdaSSP 1/100']
    #methodslist = [linreg, SSP, adaSSP_1_3, adaSSP_2_3, adaSSP_1_100, adaSSP_1_6, adaSSP_5_6]
    #gammas = [0.05 * i for i in range(1, 10)] + [0.01, 0.02, 0.03, 0.04] + [0.66, 0.95] + [0.1 ** i for i in range(3,6)] # vary by 0.05? Maybe, sure, why not. I may make this more finegrained later
    gammas  = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    #gammas = [0.02 + 0.04 * i for i in range(25)]
    #gammas = [0.01 + 0.02 * i for i in range(50)]
    #gammas = [0.05 + 0.05 * i for i in range(19)] + [0.01, 0.02, 0.03, 0.04]
    gammas.sort()
    print(f"using cross_val_splits of {cross_val_splits}, gammas are {gammas}")

    #vanilla_index = methodsNamelist.index('Non-Private')
    #if vanilla_index < 0:
        #print("You need the nonprivate version to compute relative efficiency!")
        #assert(False)

    def create_budget_func(gamma):
        def temp_func(X,y,epsilon, delta):
            return adaSSP(X=X,y=y,epsilon=epsilon,delta=delta, gamma=gamma)
        return temp_func

    methodslist = [create_budget_func(gamma) for gamma in gammas]
    methodsNamelist = [f"adaSSPbudget {gamma:.4f}" for gamma in gammas]


    #def create_budget_func_const(gamma):
        #def temp_func(X,y,epsilon, delta):
            #return constSSP(X=X,y=y,epsilon=epsilon,delta=delta, gamma=gamma)
        #return temp_func

    #constMethodslist = [create_budget_func_const(gamma) for gamma in gammas]
    #constMethodsNamelist = [f"constSSPbudget {gamma:.4f}" for gamma in gammas]

    #def create_budget_func_const_full(gamma):
        #def temp_func(X,y,epsilon, delta):
            #return constSSPfull(X=X,y=y,epsilon=epsilon,delta=delta, gamma=gamma)
        #return temp_func

    #constFullMethodslist = [create_budget_func_const_full(gamma) for gamma in gammas]
    #constFullMethodsNamelist = [f"constSSPfull {gamma:.4f}" for gamma in gammas]

    # gamma is arbitrarily set, it doesn't actually matter
    constSSPfull_func = lambda X,y,epsilon,delta: constSSPfull(X=X,y=y,epsilon=epsilon,delta=delta, gamma=0.3)


    num_n = len(nlist)
    num_eps = len(epslist)
    num_method = len(methodslist)

    # do the SSP algorithm on its own


    sigma = 1

    #theta = randn(d,1);
    theta = np.random.normal(loc=0, scale=1, size=d)
    theta_norm = np.linalg.norm(theta)
    theta = theta / (theta_norm * np.sqrt(2.0))

    non_private_results = np.zeros((num_eps, num_n))

    SSP_results = np.zeros((num_eps, num_n))
    SSP_std = np.zeros((num_eps, num_n))

    results_err = np.zeros((num_method,num_eps,num_n))
    results_std = np.zeros((num_method,num_eps,num_n))

    # for the const values (the constSSP)
    #const_results_err = np.zeros((num_method,num_eps,num_n))
    #const_results_std = np.zeros((num_method,num_eps,num_n))

    # doesn't change with gamma
    # for constSSPfull
    const_full_results_err = np.zeros((num_eps,num_n))
    const_full_results_std = np.zeros((num_eps,num_n))

    zero_error_counts = np.zeros((num_method,num_eps,num_n))

    #for i = 1:num_n
    for i in range(num_n):
        n = nlist[i]
        print(f"n is {n}")

        #X = randn(n,d);
        if correlated_data:
            pass
        else:
            X = np.random.normal(loc=0, scale=1, size=(n,d))

            #X_sqrt_sum_max = max(np.sqrt(np.sum(np.square(X), axis=1)))
            X_sqrt_sum = np.sqrt(np.sum(np.square(X), axis=1))

            #print("X_sqrt_sum is this")
            #print(X_sqrt_sum)

            #X = bsxfun(@rdivide, X,sqrt(sum(X.^2,2)));
            #https://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
            #X = X / X_sqrt_sum_max #X_sqrt_sum[:,None]
            X = X/X_sqrt_sum[:,None]


        #y = X*theta + 0.1*sigma*(rand(n,1)-0.5);
        #y = X.dot(theta) + 0.1 * sigma * (np.random.normal(loc=-0.5, scale=1, size=n))
        y = X.dot(theta) + 0.1 * sigma * (np.random.normal(loc=0, scale=1, size=n))

        #y = y - y.mean()
        #y = y / y.std()
        #cvo = cvpartition(length(y),'KFold',10);
        all_indices = np.arange(n)
        cvo = [splits for splits in ShuffleSplit(n_splits=cross_val_splits, test_size=0.1).split(all_indices)] #, random_state=0) # I could put random state here, if I wanted

        for j in range(num_eps):
        #% set parameters of the algorithm
            eps = epslist[j]
            print(f"eps is {eps:.4f}")
            #opts.eps = epslist(j);
            #opts.delta =1/n^(1.1);
            #delta = 1 / (n ** (1.1)) # not so sure what this is about
            delta = delta_func(n) # either a constant or a n^{-1.1} term

            # have to do the linreg version
            t = time.time()
            errs, cvErr,cvStd, errorDict = test_recovery(X, y, cvo, linreg, theta, eps, delta)
            assert(not np.isnan(cvErr))
            t_run = time.time() - t
            print(f"method = linreg, Time run was {t_run:.4f}s")
            non_private_results[j, i] = cvErr

            # may as well also track SSP
            t = time.time()
            errs, cvErr,cvStd, errorDict = test_recovery(X, y, cvo, SSP, theta, eps, delta)
            assert(not np.isnan(cvErr))
            t_run = time.time() - t
            print(f"method = SSP, Time run was {t_run:.4f}s")
            SSP_results[j, i] = cvErr
            SSP_std[j, i] = cvStd

            # for constSSPfull
            errs, cvErr,cvStd, errorDict = test_recovery(X, y, cvo, constSSPfull_func, theta, eps, delta)
            assert(not np.isnan(cvErr))
            const_full_results_err[j,i] = cvErr
            const_full_results_std[j,i] = cvStd



            # then for the rest of them
            for k in range(num_method):
                fun = methodslist[k]
                t = time.time()
                errs, cvErr,cvStd, errorDict = test_recovery(X, y, cvo, fun, theta, eps, delta)
                assert(not np.isnan(cvErr).any())
                #t_run=toc; # this is for timing, which I am not keeping track of
                t_run = time.time() - t
                print(f"method = {methodsNamelist[k]}, Time run was {t_run:.4f}s")
                #fprintf('%s at eps = %f: Test err = %.2f, std = %.2f, runtime = %.2f s.\n', methodsNamelist{k}, opts.eps, cvErr,cvStd,t_run)
                results_err[k,j,i] = cvErr
                results_std[k,j,i] = cvStd

                zero_error_counts[k,j,i] = errorDict["zero_counter"]

                # and for the const stuff
                #const_fun = constMethodslist[k]
                #errs, cvErr,cvStd, errorDict = test_recovery(X, y, cvo, const_fun, theta, eps, delta)
                #assert(not np.isnan(cvErr))
                #const_results_err[k,j,i] = cvErr
                #const_results_std[k,j,i] = cvStd




    #RelEfficiency =  bsxfun(@rdivide,results_err,results_err(2,:,:));
    # divide everything by the error for the vanilla linear regression

    rel_efficiency = np.zeros((num_method, num_eps, num_n))
    rel_efficiency_std = np.zeros((num_method, num_eps, num_n))

    #const_rel_efficiency = np.zeros((num_method, num_eps, num_n))
    #const_rel_efficiency_std = np.zeros((num_method, num_eps, num_n))

    const_full_rel_efficiency = np.zeros((num_eps, num_n))
    const_full_rel_efficiency_std = np.zeros((num_eps, num_n))


    SSP_rel_eff = np.zeros((num_eps, num_n))
    SSP_rel_eff_std = np.zeros((num_eps, num_n))
    vanilla_values = non_private_results #results_err[vanilla_index,:,:]


    for i in range(num_n):
        for j in range(num_eps):
            for k in range(num_method):
                rel_efficiency[k,j,i] = results_err[k,j,i] / vanilla_values[j,i]
                rel_efficiency_std[k,j,i] = results_std[k,j,i] / vanilla_values[j,i]

                #const_rel_efficiency[k,j,i] = const_results_err[k,j,i] / vanilla_values[j,i]
                #const_rel_efficiency_std[k,j,i] = const_results_std[k,j,i] / vanilla_values[j,i]

                #const_full_rel_efficiency[k,j,i] = const_full_results_err[k,j,i] / vanilla_values[j,i]
                #const_full_rel_efficiency_std[k,j,i] = const_full_results_std[k,j,i] / vanilla_values[j,i]

            SSP_rel_eff[j,i] = SSP_results[j,i] / vanilla_values[j,i]
            SSP_rel_eff_std[j,i] = SSP_std[j,i] / vanilla_values[j,i]

            const_full_rel_efficiency[j,i] = const_full_results_err[j,i] / vanilla_values[j,i]
            const_full_rel_efficiency_std[j,i] = const_full_results_std[j,i] / vanilla_values[j,i]


    # then for the plotting
    # right now, it's one plot per epsilon
    # the relative efficiency plots

    #figsize=(2*9,2*6)
    fig, axs = plt.subplots(2, 2, figsize=(16,14), sharex=True, sharey=True)
    fig.suptitle(f'Relative Efficiency vs. Gamma, {data_description} Synthetic Data', fontsize=20)
    axs_raveled = list(np.ravel(axs))

    for j in range(num_eps):
        #fig, ax = plt.subplots(1, figsize=(9,6)) # just one, for now
        ax = axs_raveled[j]
        eps = epslist[j]

        # instead of a different curve for each method, do a different curve for each n
        #for k in range(num_method):
        for i in range(num_n):

            #print("yerr is ")
            #print(np.concatenate(    (rel_efficiency_std[k,j,:].reshape((num_n,1)), np.zeros((num_n, 1))),    axis=1).reshape((2, num_n)))

            # plot gammas on the x-axis
            adaSSPbudget_yerr = np.concatenate(    (np.zeros((num_method, 1)), rel_efficiency_std[:,j,i].reshape((num_method,1))),    axis=1).transpose()
            ax.errorbar(x=gammas, y=rel_efficiency[:,j,i], yerr=adaSSPbudget_yerr, label=f"adaSSPbudget, n = {nlist[i]}")

            #temp_color = ax.get_lines()[2*i].get_color()#ax.get_color()
            # constant is 4 - for adaSSP, for SSP, constSSPfull
            temp_color = ax.get_lines()[3*i].get_color()#ax.get_color()
            #constSSPbudget_yerr = np.concatenate(    (np.zeros((num_method, 1)), const_rel_efficiency_std[:,j,i].reshape((num_method,1))),    axis=1).transpose()
            #ax.errorbar(x=gammas, y=const_rel_efficiency[:,j,i], yerr=constSSPbudget_yerr, label=f"constSSPbudget, n = {nlist[i]}", color=temp_color, linestyle="dotted")

            #constSSPfull_yerr = np.concatenate(    (np.zeros((num_method, 1)), const_full_rel_efficiency_std[:,j,i].reshape((num_method,1))),    axis=1).transpose()

            constSSPfull_vals = [const_full_rel_efficiency[j,i]] * len(gammas)
            constSSPfull_stds = np.array([const_full_rel_efficiency_std[j,i]] * len(gammas))
            constSSPfull_yerr = np.concatenate(    (np.zeros((num_method, 1)), constSSPfull_stds.reshape((num_method,1))),    axis=1).transpose()
            ax.errorbar(x=gammas, y=constSSPfull_vals, yerr=constSSPfull_yerr, label=f"constSSPfull, n = {nlist[i]}", color=temp_color, linestyle="dotted")

            #ax.errorbar(x=gammas, y=const_full_rel_efficiency[:,j,i], yerr=constSSPfull_yerr, label=f"constSSPfull, n = {nlist[i]}", color=temp_color, linestyle="dashdot")

            ssp_vals = [SSP_rel_eff[j,i]] * len(gammas)
            ssp_stds = np.array([SSP_rel_eff_std[j,i]] * len(gammas))
            #ax.plot([0,1], [SSP_rel_eff[j,i], SSP_rel_eff[j,i]], label=f"SSP, n = {nlist[i]}", color=temp_color, linestyle="dashed")

            ax.errorbar(x=gammas, y=ssp_vals, yerr=np.concatenate(    (np.zeros((num_method, 1)), ssp_stds.reshape((num_method,1))),    axis=1).transpose(), label=f"SSP, n = {nlist[i]}", color=temp_color, linestyle="dashed")

        ax.plot([1.0/3.0, 1.0/3.0], [10**(-1), 10** 9], label="adaSSP gamma value",color="red")
        ax.plot([0, 1.0], [1.0, 1.0], label="non-private linear regression", color="purple")
        #ax.set_xlabel("n values (size of data set)")
        ax.set_xlabel("Gamma")
        ax.set_ylabel('Relative Efficiency')
        #ax.set_title(f"Relative Efficiency For epsilon = {eps:.3f}, Synthetic Data")
        ax.set_title(f"epsilon = {eps:.3f}")
        ax.set_yscale('log')
        #ax.set_xscale('log')
        #ax.legend()
        #plt.savefig(f"plots/rel_eff_by_gamma_eps_{eps:.4f}_synthetic.png")
    #for ax in fig.get_axes():
        #ax.label_outer()

    # intentionally calling on last axis
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"plots/rel_eff_by_gamma_all_eps_{data_description}_synthetic.png")
    plt.close()



    fig, axs = plt.subplots(2, 2, figsize=(16,14), sharex=True, sharey=True)
    fig.suptitle(f'Lambda Zero Proportion for adaSSPbudget, {data_description} Synthetic Data', fontsize=20)
    axs_raveled = list(np.ravel(axs))

    # for plotting zeroes
    for j in range(num_eps):
        #fig, ax = plt.subplots(1, figsize=(9,6)) # just one, for now
        eps = epslist[j]
        ax = axs_raveled[j]

        # instead of a different curve for each method, do a different curve for each n
        #for k in range(num_method):
        for i in range(num_n):

            #print("yerr is ")
            #print(np.concatenate(    (rel_efficiency_std[k,j,:].reshape((num_n,1)), np.zeros((num_n, 1))),    axis=1).reshape((2, num_n)))

            # plot gammas on the x-axis
            # I may want to investigate ways to avoid overlapping plots
            zero_props = [val / cross_val_splits for val in list(zero_error_counts[:,j,i])]
            # if you are estimating the sum of indicators over n, the variance is p(1-p)/n, so use estimate of p
            zero_props_stds = [np.sqrt(p * (1-p) / cross_val_splits) for p in zero_props]
            ax.errorbar(x=gammas, y=zero_props, yerr=zero_props_stds, label=f"n = {nlist[i]}", alpha=0.7)
            #ax.plot(gammas,error_props, label=f"n = {nlist[i]}", alpha=0.7)

        ax.set_xlabel("Gamma")
        ax.set_ylabel('Lambda Zero Proportion for adaSSPbudget')
        #ax.set_title(f"Lambda Zero Proportion For epsilon = {eps:.3f}, Synthetic Data")
        ax.set_title(f"epsilon = {eps:.3f}")
        #ax.set_yscale('log')
        #ax.set_xscale('log')
        #ax.legend()
        #plt.savefig(f"plots/zero_count_by_gamma_eps_{eps:.4f}_synthetic.png")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"plots/zero_count_by_gamma_all_eps_synthetic.png")
    plt.close()


def plots_by_gamma_pred_err_synth():
    d = 10
    epslist = [0.001, 0.01, 0.1,1]
    #epslist = [0.01, 0.1]
    #nlist= 20*2.^[1:2:18];
    #nlist = [20 * (2 ** i) for i in range(1,16,2)] # smaller n go faster

    #nlist = [20 * (2 ** i) for i in [6, 10, 14, 18]] # maybe have 18... # start small
    nlist = [20 * (2 ** i) for i in [6, 10, 14]]
    #cross_val_splits = 32 # maybe crank up for real results...
    cross_val_splits=6
    #gammas = [0.05 * i for i in range(1, 10)] + [0.01, 0.02, 0.03, 0.04] + [0.66, 0.95] + [0.1 ** i for i in range(3,6)] # vary by 0.05? Maybe, sure, why not. I may make this more finegrained later
    gammas  = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    #gammas = [0.01 + 0.02 * i for i in range(50)]
    #gammas = [0.05 + 0.05 * i for i in range(19)] + [0.01, 0.02, 0.03, 0.04]
    gammas.sort()
    print(f"using cross_val_splits of {cross_val_splits}, gammas are {gammas}")

    #vanilla_index = methodsNamelist.index('Non-Private')
    #if vanilla_index < 0:
        #print("You need the nonprivate version to compute relative efficiency!")
        #assert(False)

    def create_budget_func(gamma):
        def temp_func(X,y,epsilon, delta):
            return adaSSP(X=X,y=y,epsilon=epsilon,delta=delta, gamma=gamma)
        return temp_func

    methodslist = [create_budget_func(gamma) for gamma in gammas]
    methodsNamelist = [f"adaSSPbudget {gamma:.4f}" for gamma in gammas]


    #def create_budget_func_const(gamma):
        #def temp_func(X,y,epsilon, delta):
        #    return constSSP(X=X,y=y,epsilon=epsilon,delta=delta, gamma=gamma)
        #return temp_func

    #constMethodslist = [create_budget_func_const(gamma) for gamma in gammas]
    #constMethodsNamelist = [f"constSSPbudget {gamma:.4f}" for gamma in gammas]

    #def create_budget_func_const_full(gamma):
    #    def temp_func(X,y,epsilon, delta):
    #        return constSSPfull(X=X,y=y,epsilon=epsilon,delta=delta, gamma=gamma)
    #    return temp_func

    constSSPfull_func = lambda X,y,epsilon,delta: constSSPfull(X=X,y=y,epsilon=epsilon,delta=delta, gamma=0.3)


    #constFullMethodslist = [create_budget_func_const_full(gamma) for gamma in gammas]
    #constFullMethodsNamelist = [f"constSSPfull {gamma:.4f}" for gamma in gammas]


    num_n = len(nlist)
    num_eps = len(epslist)
    num_method = len(methodslist)

    # do the SSP algorithm on its own


    sigma = 1

    #theta = randn(d,1);
    theta = np.random.normal(loc=0, scale=1, size=d)
    theta_norm = np.linalg.norm(theta)
    theta = theta / (theta_norm * np.sqrt(2.0))

    non_private_results = np.zeros((num_eps, num_n))
    SSP_results = np.zeros((num_eps, num_n))
    SSP_std = np.zeros((num_eps, num_n))

    results_err = np.zeros((num_method,num_eps,num_n))
    results_std = np.zeros((num_method,num_eps,num_n))

    # for the const values (the constSSP)
    #const_results_err = np.zeros((num_method,num_eps,num_n))
    #const_results_std = np.zeros((num_method,num_eps,num_n))

    const_full_results_err = np.zeros((num_eps,num_n))
    const_full_results_std = np.zeros((num_eps,num_n))

    zero_error_counts = np.zeros((num_method,num_eps,num_n))

    #for i = 1:num_n
    for i in range(num_n):
        n = nlist[i]
        print(f"n is {n}")

        #X = randn(n,d);
        X = np.random.normal(loc=0, scale=1, size=(n,d))

        X_sqrt_sum = np.sqrt(np.sum(np.square(X), axis=1))

        #X = bsxfun(@rdivide, X,sqrt(sum(X.^2,2)));
        #https://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
        X = X / X_sqrt_sum[:,None]
        #y = X*theta + 0.1*sigma*(rand(n,1)-0.5);
        #y = X.dot(theta) + 0.1 * sigma * (np.random.normal(loc=-0.5, scale=1, size=n))
        # removing the negatie sign shouldn't be a big deal, right?
        y = X.dot(theta) + 0.1 * sigma * (np.random.normal(loc=0, scale=1, size=n))

        #print("Make sure you know what you're doing with standardizing y")
        #y = y - y.mean()
        #y = y / y.std()
        #cvo = cvpartition(length(y),'KFold',10);
        all_indices = np.arange(n)
        cvo = [splits for splits in ShuffleSplit(n_splits=cross_val_splits, test_size=0.1).split(all_indices)] #, random_state=0) # I could put random state here, if I wanted


        for j in range(num_eps):
        #% set parameters of the algorithm
            eps = epslist[j]
            print(f"eps is {eps:.4f}")
            #opts.eps = epslist(j);
            #opts.delta =1/n^(1.1);
            #delta = 1 / (n ** (1.1)) # not so sure what this is about
            delta = delta_func(n) # either a constant or a n^{-1.1} term

            # have to do the linreg version
            t = time.time()
            errs, cvErr,cvStd, errorDict = test_prediction_error(X, y, cvo, linreg, eps, delta, d)
            assert(not np.isnan(cvErr))
            t_run = time.time() - t
            print(f"method = linreg, Time run was {t_run:.4f}s")
            non_private_results[j, i] = cvErr

            # may as well also track SSP
            t = time.time()
            errs, cvErr,cvStd, errorDict = test_prediction_error(X, y, cvo, SSP, eps, delta, d)
            assert(not np.isnan(cvErr))
            t_run = time.time() - t
            print(f"method = SSP, Time run was {t_run:.4f}s")
            SSP_results[j, i] = cvErr
            SSP_std[j, i] = cvStd

            #const_full_fun = constFullMethodslist[k]
            errs, cvErr,cvStd, errorDict = test_prediction_error(X, y, cvo, constSSPfull_func, eps, delta, d)
            assert(not np.isnan(cvErr))
            const_full_results_err[j,i] = cvErr
            const_full_results_std[j,i] = cvStd

            # then for the rest of them
            for k in range(num_method):
                fun = methodslist[k]
                t = time.time()
                errs, cvErr,cvStd, errorDict = test_prediction_error(X, y, cvo, fun, eps, delta, d)
                assert(not np.isnan(cvErr).any())
                #t_run=toc; # this is for timing, which I am not keeping track of
                t_run = time.time() - t
                print(f"method = {methodsNamelist[k]}, Time run was {t_run:.4f}s")
                #fprintf('%s at eps = %f: Test err = %.2f, std = %.2f, runtime = %.2f s.\n', methodsNamelist{k}, opts.eps, cvErr,cvStd,t_run)
                results_err[k,j,i] = cvErr
                results_std[k,j,i] = cvStd

                zero_error_counts[k,j,i] = errorDict["zero_counter"]

                # and for the const stuff
                #const_fun = constMethodslist[k]
                #errs, cvErr,cvStd, errorDict = test_prediction_error(X, y, cvo, const_fun, eps, delta, d)
                #assert(not np.isnan(cvErr))
                #const_results_err[k,j,i] = cvErr
                #const_results_std[k,j,i] = cvStd




    # plotting prediction error
    fig, axs = plt.subplots(2, 2, figsize=(16,14), sharex=True, sharey=True)
    fig.suptitle('Prediction Error (MSE), Synthetic Data', fontsize=20)
    axs_raveled = list(np.ravel(axs))
    for j in range(num_eps):
        #fig, ax = plt.subplots(1, figsize=(9,6)) # just one, for now
        eps = epslist[j]
        ax = axs_raveled[j]

        # instead of a different curve for each method, do a different curve for each n
        #for k in range(num_method):
        for i in range(num_n):

            #print("yerr is ")

            # plot gammas on the x-axis
            adaSSPbudget_yerr = np.concatenate(    (np.zeros((num_method, 1)), results_std[:,j,i].reshape((num_method,1))),    axis=1).transpose()
            ax.errorbar(x=gammas, y=results_err[:,j,i], yerr=adaSSPbudget_yerr, label=f"adaSSPbudget, n={nlist[i]}")

            #constant set to 4 because of SSP, adaSSPbudget, constSSPfull
            temp_color = ax.get_lines()[3*i].get_color()

            #constSSPbudget_yerr = np.concatenate(    (np.zeros((num_method, 1)), const_results_std[:,j,i].reshape((num_method,1))),    axis=1).transpose()
            #ax.errorbar(x=gammas, y=const_results_err[:,j,i], yerr=constSSPbudget_yerr, label=f"constSSPbudget, n={nlist[i]}", color=temp_color, linestyle="dotted")

            #constSSPfull_yerr = np.concatenate(    (np.zeros((num_method, 1)), const_full_results_std[:,j,i].reshape((num_method,1))),    axis=1).transpose()
            #ax.errorbar(x=gammas, y=const_full_results_err[:,j,i], yerr=constSSPfull_yerr, label=f"constSSPfull, n={nlist[i]}", color=temp_color, linestyle="dashdot")

            constSSPfull_vals = [const_full_results_err[j,i]] * len(gammas)
            constSSPfull_stds = np.array([const_full_results_std[j,i]] * len(gammas))
            constSSPfull_yerr = np.concatenate(    (np.zeros((num_method, 1)), constSSPfull_stds.reshape((num_method,1))),    axis=1).transpose()
            ax.errorbar(x=gammas, y=constSSPfull_vals, yerr=constSSPfull_yerr, label=f"constSSPfull, n = {nlist[i]}", color=temp_color, linestyle="dotted")


            ssp_vals = [SSP_results[j,i]] * len(gammas)
            ssp_stds = np.array([SSP_std[j,i]] * len(gammas))
            #ax.plot([0,1], [SSP_rel_eff[j,i], SSP_rel_eff[j,i]], label=f"SSP, n = {nlist[i]}", color=temp_color, linestyle="dashed")

            ax.errorbar(x=gammas, y=ssp_vals, yerr=np.concatenate(    (np.zeros((num_method, 1)), ssp_stds.reshape((num_method,1))),    axis=1).transpose(), label=f"SSP, n={nlist[i]}", color=temp_color, linestyle="dashed")

        ax.plot([1.0/3.0, 1.0/3.0], [0, 10** 4], label="adaSSP gamma value")
        ax.plot([0.0, 1.0], [1.0, 1.0], label="baseline model predicting mean")
        #ax.set_xlabel("n values (size of data set)")
        ax.set_xlabel("Gamma")
        ax.set_ylabel('Prediction Error (MSE)')
        #ax.set_title(f"Prediction Error For epsilon = {eps:.3f}, Synthetic Data")
        ax.set_title(f"epsilon = {eps:.3f}")
        ax.set_yscale('log')
        #ax.set_xscale('log')
        #ax.legend()
        #plt.savefig(f"plots/pred_err_by_gamma_eps_{eps:.4f}_synthetic.png")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"plots/pred_err_by_gamma_all_eps_synthetic.png")
    plt.close()


"""
# this checks non private ridge regression
def non_private_checks():
    d = 10
    epslist = [0.01, 0.1,1]
    #epslist = [0.01, 0.1]
    #nlist= 20*2.^[1:2:18];
    #nlist = [20 * (2 ** i) for i in range(1,16,2)] # smaller n go faster

    #nlist = [20 * (2 ** i) for i in [6, 10, 14, 18]] # maybe have 18... # start small
    nlist = [20 * (2 ** i) for i in [6, 10, 14]]
    cross_val_splits = 6 # maybe crank up for real results...


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


    num_n = len(nlist)
    num_method = len(methodslist)

    # do the SSP algorithm on its own


    sigma = 1

    #theta = randn(d,1);
    theta = np.random.normal(loc=0, scale=1, size=d)
    theta_norm = np.linalg.norm(theta)
    theta = theta / (theta_norm * np.sqrt(2.0))

    linreg_results = np.zeros((num_n))
    linreg_std = np.zeros((num_n))

    ridge_results = np.zeros((num_method, num_n))
    ridge_std = np.zeros((num_method, num_n))

    #SSP_results = np.zeros((num_eps, num_n))
    #SSP_std = np.zeros((num_eps, num_n))


    zero_error_counts = np.zeros((num_method,num_n))

    #for i = 1:num_n
    for i in range(num_n):
        n = nlist[i]
        print(f"n is {n}")

        #X = randn(n,d);
        X = np.random.normal(loc=0, scale=1, size=(n,d))

        X_sqrt_sum = np.sqrt(np.sum(np.square(X), axis=1))

        #X = bsxfun(@rdivide, X,sqrt(sum(X.^2,2)));
        #https://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
        X = X / X_sqrt_sum[:,None]
        y = X.dot(theta) + 0.1 * sigma * (np.random.normal(loc=-0.5, scale=1, size=n))


        all_indices = np.arange(n)
        cvo = [splits for splits in ShuffleSplit(n_splits=cross_val_splits, test_size=0.1).split(all_indices)] #, random_state=0) # I could put random state here, if I wanted

        delta = 1 / (n ** (1.1)) # not so sure what this is about
        eps = 1.0

        # have to do the linreg version
        t = time.time()
        errs, cvErr,cvStd, errorDict = test_recovery(X, y, cvo, linreg, theta, eps, delta)
        assert(not np.isnan(cvErr))
        t_run = time.time() - t
        print(f"method = linreg, Time run was {t_run:.4f}s")
        linreg_results[i] = cvErr
        linreg_std[i] = cvStd

        # then for the rest of them
        for k in range(num_method):
            fun = methodslist[k]
            t = time.time()
            errs, cvErr,cvStd, errorDict = test_recovery(X, y, cvo, fun, theta, eps, delta)
            assert(not np.isnan(cvErr))
            #t_run=toc; # this is for timing, which I am not keeping track of
            t_run = time.time() - t
            print(f"method = {methodsNamelist[k]}, Time run was {t_run:.4f}s")
            #fprintf('%s at eps = %f: Test err = %.2f, std = %.2f, runtime = %.2f s.\n', methodsNamelist{k}, opts.eps, cvErr,cvStd,t_run)
            ridge_results[k,i] = cvErr
            ridge_std[k,i] = cvStd





    # maybe save('exp_gaussian.mat','results_err','results_std')

    #RelEfficiency =  bsxfun(@rdivide,results_err,results_err(2,:,:));
    # divide everything by the error for the vanilla linear regression

    rel_efficiency = np.zeros((num_method, num_n))
    rel_efficiency_std = np.zeros((num_method, num_n))

    vanilla_values =  linreg_results # lin_results #results_err[vanilla_index,:,:]

    linreg_relative_stds = np.zeros((num_n))


    for i in range(num_n):
        for k in range(num_method):
            rel_efficiency[k,i] = ridge_results[k,i] / vanilla_values[i]
            rel_efficiency_std[k,i] = ridge_std[k,i] / vanilla_values[i]

        linreg_relative_stds[i] = linreg_std[i] / vanilla_values[i]





    # then for the plotting
    # right now, it's one plot per epsilon
    # the relative efficiency plots

    fig, ax = plt.subplots(1, figsize=(9,6)) # just one, for now


    # instead of a different curve for each method, do a different curve for each n
    #for k in range(num_method):
    for i in range(num_n):

        #print("yerr is ")
        #print(np.concatenate(    (rel_efficiency_std[k,j,:].reshape((num_n,1)), np.zeros((num_n, 1))),    axis=1).reshape((2, num_n)))

        # plot gammas on the x-axis
        ax.errorbar(x=lambs, y=rel_efficiency[:,i], yerr=np.concatenate(    (np.zeros((num_method, 1)), rel_efficiency_std[:,i].reshape((num_method,1))),    axis=1).transpose(), label=f"n = {nlist[i]}")

        temp_color = ax.get_lines()[2*i].get_color()#ax.get_color()

        #ssp_vals = [SSP_rel_eff[j,i]] * len(gammas)
        #ssp_stds = np.array([SSP_rel_eff_std[j,i]] * len(gammas))
        linreg_vals_display = [1.0] * len(lambs)
        linreg_stds_display = np.array([linreg_relative_stds[i]] * len(lambs))
        #ax.plot([0,1], [SSP_rel_eff[j,i], SSP_rel_eff[j,i]], label=f"SSP, n = {nlist[i]}", color=temp_color, linestyle="dashed")

        ax.errorbar(x=lambs, y=linreg_vals_display, yerr=np.concatenate(    (np.zeros((num_method, 1)), linreg_stds_display.reshape((num_method,1))),    axis=1).transpose(), label=f"linreg, n = {nlist[i]}", color=temp_color, linestyle="dashed")

    #ax.set_xlabel("n values (size of data set)")
    ax.set_xlabel("Ridge Regression Lambda Parameter")
    ax.set_ylabel('Relative Efficiency')
    ax.set_title(f"Relative Efficiency, Synthetic Data, Nonprivate Methods (Ridge and Linear Regression)")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    plt.savefig(f"plots/nonprivate_synthetic.png")
"""


if __name__ == "__main__":
    #plots_by_n()
    plots_by_gamma()
    #plots_by_gamma_pred_err_synth()
    #non_private_checks()

