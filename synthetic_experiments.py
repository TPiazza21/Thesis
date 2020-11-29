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
from linreg import linreg
from adassp import adaSSP, adaSSP_1_3, adaSSP_2_3

from test_recovery import test_recovery



# I may want to set a seed early on, so I have reproducible results


d = 10
epslist = [0.01,0.1,1]
#nlist= 20*2.^[1:2:18];
nlist = [20 * (2 ** i) for i in range(1,17,2)] # smaller n go faster

cross_val_splits = 6



#methodslist = [@suffstats_perturb,@adassp, @budget_adassp_1_6, @budget_adassp_1_3,...
    #@budget_adassp_1_2, @budget_adassp_2_3, @budget_adassp_5_6, @budget_adassp_1_100]
methodsNamelist = ['Non-Private', 'SSP', 'Budget AdaSSP 1/3', 'Budget AdaSSP 2/3'] #, 'Budget AdaSSP 1/6', 'Budget AdaSSP 1/3', 'Budget AdaSSP 1/2',
#'Budget AdaSSP 2/3', 'Budget AdaSSP 5/6', 'Budget AdaSSP 1/100']
methodslist = [linreg, SSP, adaSSP_1_3, adaSSP_2_3]

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
results_time = np.zeros((num_method,num_eps,num_n))

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
    y = X.dot(theta) + 0.1 * sigma * (np.random.normal(loc=-0.5, scale=1, size=n))

    #cvo = cvpartition(length(y),'KFold',10);
    all_indices = np.arange(n)
    cvo = [splits for splits in ShuffleSplit(n_splits=cross_val_splits, test_size=0.1).split(all_indices)] #, random_state=0) # I could put random state here, if I wanted


    for j in range(num_eps):
    #% set parameters of the algorithm
        eps = epslist[j]
        #opts.eps = epslist(j);
        #opts.delta =1/n^(1.1);
        delta = 1 / (n ** (1.1)) # not so sure what this is about

        for k in range(num_method):
            fun = methodslist[k]
            t = time.time()
            errs, cvErr,cvStd = test_recovery(X, y, cvo, fun, theta, eps, delta)
            assert(not np.isnan(cvErr))
            #t_run=toc; # this is for timing, which I am not keeping track of
            t_run = time.time() - t
            print(f"Time run was {t_run:.4f}s")
            #fprintf('%s at eps = %f: Test err = %.2f, std = %.2f, runtime = %.2f s.\n', methodsNamelist{k}, opts.eps, cvErr,cvStd,t_run)
            results_err[k,j,i] = cvErr
            results_std[k,j,i] = cvStd
            results_time[k,j,i] = t_run


# maybe save('exp_gaussian.mat','results_err','results_std')

#RelEfficiency =  bsxfun(@rdivide,results_err,results_err(2,:,:));
# divide everything by the error for the vanilla linear regression

rel_efficiency = np.zeros((num_method, num_eps, num_n))
vanilla_values = results_err[vanilla_index,:,:]

for i in range(num_n):
    for j in range(num_eps):
        for k in range(num_method):
            rel_efficiency[k,j,i] = results_err[k,j,i] / vanilla_values[j,i]

# then for the plotting
# right now, it's one plot per epsilon
# the relative efficiency plots

for j in range(num_eps):
    fig, ax = plt.subplots(1) # just one, for now
    eps = epslist[j]

    for k in range(num_method):
        ax.errorbar(x=nlist, y=rel_efficiency[k,j,:], yerr=results_std[k,j,:], label=methodsNamelist[k])

    ax.set_xlabel("n values (size of data set)")
    ax.set_ylabel('Relative Efficiency')
    ax.set_title(f"Relative Effieincy For epsilon = {eps:.2f}, Synthetic Data")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    plt.savefig(f"plots/rel_eff_eps_{eps:.4f}_synthetic.png")


# and if we don't divide the values
for j in range(num_eps):
    fig, ax = plt.subplots(1) # just one, for now
    eps = epslist[j]

    for k in range(num_method):

        ax.errorbar(x=nlist, y=results_err[k,j,:], yerr=results_std[k,j,:], label=methodsNamelist[k])

    ax.set_xlabel("n values (size of data set)")
    ax.set_ylabel('E (theta_true-theta_pred)^2')
    ax.set_title(f"E (theta_true-theta_pred)^2 for epsilon = {eps:.2f}, Synthetic Data")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    plt.savefig(f"plots/theta_dist_eps_{eps:.4f}_synthetic.png")
