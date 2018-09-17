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

    X = torch.tensor(series)
    X = X.type(torch.DoubleTensor)
    X_1 = shift_tensor(X, 1)
    data_tensors = []

    X = X.narrow(0, 0, X.shape[0] - 1)
    dX = X_1 - X

    for i in range(1, n + 1):
        data_tensors.(dX.narrow(0, n - i, (dX.shape[0] - n)))

    data_tensors[0] = torch.reshape(data_tensors[0], (data_tensors[0].shape[0], 1))
    for i in range(1, n):
        data_tensors[i] = torch.reshape(data_tensors[i], (data_tensors[0].shape[0], 1))
        data_tensors[0] = torch.cat((data_tensors[0], data_tensors[i]), 1)

    X = X.narrow(0, 0, X.shape[0] - n)
    dX = dX.narrow(0, n, dX.shape[0] - n)

    X = torch.cat((torch.reshape(X, (X.shape[0], 1)), data_tensors[0]), 1)
    dX = torch.reshape(dX, (dX.shape[0], 1))

    on = torch.ones((X.shape[0], 1))

    X_ = torch.cat((X, torch.ones_like(on, dtype=torch.float64)), 1)

    nobs = X_.shape[0]

    # Xb = y -> Xt.X.b = Xt.y -> b = (Xt.X)^-1.Xt.y
    coeff = torch.mm(torch.mm(torch.inverse(torch.mm(torch.t(X_), X_)), torch.t(X_)), dX)
    coeff_std_err = get_coeff_std_error(X_, get_std_error(X_, dX, coeff), coeff)[0]
    t_stat = coeff[0]/coeff_std_err

    p = mackinnonp(t_stat.item(), regression="c", N=1)
    critvalues = mackinnoncrit(N=1, regression="c", nobs=nobs)
    critvalues = {"1%" : critvalues[0], "5%" : critvalues[1], "10%" : critvalues[2]}

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
