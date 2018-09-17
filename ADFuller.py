#coding:utf-8
"""Augmented Dickey-Fuller test implemented using Pytorch"""
import math
import torch
from statsmodels.tsa.adfvalues import mackinnonp, mackinnoncrit

def ad_fuller(series, maxlag=None):
    """Get series and return the p-value and the t-stat of the coefficient"""
    if maxlag is None:
        n = int((len(series) - 1) ** (1./3))
    elif maxlag < 1:
        n = 1
    else:
        n = maxlag

    X = torch.tensor(series)        # Putting the X values on a Tensor
    X = X.type(torch.DoubleTensor)  # with Double as type.

    X_1 = shift_tensor(X, 1)        # Generating the lagged tensor to calculate the difference
    data_tensors = []

    X = X.narrow(0, 0, X.shape[0] - 1) # Re-sizing the x values to get the difference
    dX = X_1 - X                       # Calculating the difference

    # Storing the lagged difference tensors
    for i in range(1, n + 1):
        data_tensors.append(dX.narrow(0, n - i, (dX.shape[0] - n))) 

    # Concatenating the lagged tensors into a single one
    lagged_tensors = torch.reshape(data_tensors[0], (data_tensors[0].shape[0], 1))
    for i in range(1, n):
        data_tensors[i] = torch.reshape(data_tensors[i], (lagged_tensors.shape[0], 1))
        lagged_tensors = torch.cat((lagged_tensors, data_tensors[i]), 1)

    # Reshaping the X and the difference tensor to match the dimension of the lagged ones
    X = X.narrow(0, 0, X.shape[0] - n)
    dX = dX.narrow(0, n, dX.shape[0] - n)
    dX = torch.reshape(dX, (dX.shape[0], 1))

    # Concatenating the lagged tensors to the X one 
    # and adding a column full of ones for the Linear Regression
    X = torch.cat((torch.reshape(X, (X.shape[0], 1)), lagged_tensors), 1)
    on = torch.ones((X.shape[0], 1))
    X_ = torch.cat((X, torch.ones_like(on, dtype=torch.float64)), 1)

    nobs = X_.shape[0]

    # Xb = y -> Xt.X.b = Xt.y -> b = (Xt.X)^-1.Xt.y
    coeff = torch.mm(torch.mm(torch.inverse(torch.mm(torch.t(X_), X_)), torch.t(X_)), dX)

    # Get the coefficients std error and then the t-stat
    coeff_std_err = get_coeff_std_error(X_, get_std_error(X_, dX, coeff), coeff)[0]
    t_stat = coeff[0]/coeff_std_err

    # With the t-stat get the p-value for the time series
    p = mackinnonp(t_stat.item(), regression="c", N=1)
    critvalues = mackinnoncrit(N=1, regression="c", nobs=nobs)
    critvalues = {"1%" : critvalues[0], "5%" : critvalues[1], "10%" : critvalues[2]}

    # Returning a tuple on the statsmodels format
    return t_stat.item(), p.item(), n, nobs, critvalues

def get_coeff_std_error(X, std_error, p):
    """Receive the regression standard error 
    and calculate for the coefficient p"""
    std_coeff = []
    for i in range(len(p)):
        s = torch.inverse(torch.mm(torch.t(X), X))[i][i] * (std_error ** 2)
        s = math.sqrt(s.item())
        std_coeff.append(s)
    return std_coeff

def get_std_error(X, label, p):
    """Get the regression standard error"""
    std_error = 0
    y_new = torch.mm(X, p)
    for i in range(len(X)):
        diff = (label[i][0] - y_new[i][0]) ** 2
        std_error += diff.item()
    std_error = math.sqrt(std_error/X.shape[0])

    return std_error

def shift_tensor(t, shift):
    """Shift the tensor 1 unit to calculate the difference"""
    if shift == 0:
        return t

    return t.narrow(0, shift, t.shape[0] - shift)
