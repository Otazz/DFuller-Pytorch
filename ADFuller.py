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

    # Putting the X values on a Tensor with Double as type
    X = torch.tensor(series) 
    X = X.type(torch.DoubleTensor)

    # Generating the lagged tensor to calculate the difference
    X_1 = shift_tensor(X, 1)

    # Re-sizing the x values to get the difference
    X = X.narrow(0, 0, X.shape[0] - 1)
    dX = X_1 - X

    # Generating the lagged difference tensors
    # and concatenating the lagged tensors into a single one
    for i in range(1, n + 1):
        lagged_n = dX.narrow(0, n - i, (dX.shape[0] - n))
        lagged_reshape = torch.reshape(lagged_n, (lagged_n.shape[0], 1))
        if i == 1:
            lagged_tensors = lagged_reshape
        else:
            lagged_tensors = torch.cat((lagged_tensors, lagged_reshape), 1)

    # Reshaping the X and the difference tensor 
    # to match the dimension of the lagged ones
    X = X.narrow(0, 0, X.shape[0] - n)
    dX = dX.narrow(0, n, dX.shape[0] - n)
    dX = torch.reshape(dX, (dX.shape[0], 1))

    # Concatenating the lagged tensors to the X one
    # and adding a column full of ones for the Linear Regression
    X = torch.cat((torch.reshape(X, (X.shape[0], 1)), lagged_tensors), 1)
    ones_columns = torch.ones((X.shape[0], 1))
    X_ = torch.cat((X, torch.ones_like(ones_columns, dtype=torch.float64)), 1)

    nobs = X_.shape[0]

    # Xb = y -> Xt.X.b = Xt.y -> b = (Xt.X)^-1.Xt.y
    coeff = torch.mm(torch.mm(torch.inverse(
            torch.mm(torch.t(X_), X_)), torch.t(X_)), dX)

    std_error = get_std_error(X_, dX, coeff)
    coeff_std_err = get_coeff_std_error(X_, std_error, coeff)[0]
    t_stat = coeff[0]/coeff_std_err

    p_value = mackinnonp(t_stat.item(), regression="c", N=1)
    critvalues = mackinnoncrit(N=1, regression="c", nobs=nobs)
    critvalues = {
                  "1%" : critvalues[0], 
                  "5%" : critvalues[1], 
                  "10%" : critvalues[2]
                 }

    return t_stat.item(), p_value, n, nobs, critvalues

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

def shift_tensor(tensor, shift):
    """Shift the tensor 1 unit to calculate the difference"""
    if shift == 0:
        return tensor

    return tensor.narrow(0, shift, tensor.shape[0] - shift)
