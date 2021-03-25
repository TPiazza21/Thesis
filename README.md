# Thesis for Tyler Piazza, Harvard Class of 2021
# Differentially Private Ridge Regression: The Cost of a Hyperparameter
## Thesis Advisor was Professor Salil Vadhan

### Some of this code (definition of SSP and adaSSP, and generation of synthetic and UCI data) was adapted from the MATLAB code from https://arxiv.org/pdf/1803.02596.pdf, via the Github link https://github.com/yuxiangw/optimal_dp_linear_regression.

## For each of the files

### synthetic_experiments.py
- This is where the synthetic data was generated, and prediction error and relative efficiency (and lambda zero proportion) are here. Note that synthetic_experiments_second_run.py can be ignored - it should be largely the same as synthetic_experiments.py, it was created just to run slightly different programs at the same time

### data_experiments.py
- UCI data is parsed and then prediction error was plotted here

### SSP.py
- The definition of SSP, Sufficient Statistics Perturbation. The baseline differentially private mechanism

### adassp.py
- has definitions of adaSSP, adaSSPbudget, constSSPfull, and the remnants of constSSP (also known as constSSPbudget)
- also has some hardcoded gamma values in place of adaSSPbudget

### linreg.py
- where (non-private) linear regression and ridge regression are implemented. Beware of adding identity matrices to the XTX matrix!

### test_recovery.py
- where relative efficiency and prediction error are implemented
