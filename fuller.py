#coding:utf-8
"""Dickey-Fuller test implemented using Pytorch"""
import math
import torch

#import if wanted to plot the regression
#from matplotlib import pyplot

def d_fuller(series):
    """Get series and return the p-value and the t-stat of the coefficient"""
    X = torch.tensor(series)
    X = X.type(torch.DoubleTensor)
    X_1 = shift_tensor(X, 1)

    X = torch.narrow(X, 0, 0, X.shape[0] - 1)
    dX = X_1 - X

    X = torch.reshape(X, (X.shape[0], 1))
    dX = torch.reshape(dX, (dX.shape[0], 1))

    on = torch.ones(X.shape)
    X_ = torch.cat((X, torch.ones_like(on, dtype=torch.float64)), 1)

    coeff = torch.mm(torch.mm(torch.inverse(torch.mm(torch.t(X_), X_)), 
                                                     torch.t(X_)), dX)
    std_err = get_std_error(X_, dX, coeff)
    coeff_std_err = get_coeff_std_error(X_, std_err, coeff)

    t_stat_1st_coeff = coeff[0]/coeff_std_err[0]

    yhat = torch.mm(X_, coeff)
    p  = coeff[0].item() + 1
    t_stat = t_stat_1st_coeff.item()

    # plot data and predictions
    #pyplot.scatter(X, dX)
    #pyplot.plot(X.data.numpy(), yhat.data.numpy(), color='red')
    #pyplot.show()

    return p, t_stat


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
    """Shift the tensor to calculate the difference"""
    if shift == 0:
        return t

    return t.narrow(0, shift, t.shape[0] - shift)

def get_conclusion(t_stat, n):
    """VALUES NOT ALWAYS CORRECT
         returns a string based on rejection of the null hypothesis"""
    df_critical = {
        "percentages" : [1, 2.5, 5, 10, 90, 95, 97.5, 99],
        25    : [-3.75, -3.33, -3.00, -2.62, -0.37, 0.00,  0.34, 0.72],
        50    : [-3.58, -3.22, -2.93, -2.60, -0.40, -0.03, 0.29, 0.66],
        100   : [-3.51, -3.17, -2.89, -2.58, -0.42, -0.05, 0.26, 0.63],
        250   : [-3.46, -3.14, -2.88, -2.57, -0.42, -0.06, 0.24, 0.62],
        500   : [-3.44, -3.13, -2.87, -2.57, -0.43, -0.07, 0.24, 0.61],
        "more": [-3.43, -3.12, -2.86, -2.57, -0.44, -0.07, 0.23, 0.60]
    }

    index = -1
    key = "more"
    if n <= 25:
        key = 25
    elif n <= 50:
        key = 50
    elif n <= 100:
        key = 100
    elif n <= 500:
        key = 500

    nums = df_critical[key]
    for i in range(len(nums)):
        if t_stat < nums[i]:
            index = i
            break

    if index == -1:
        return "The series is not stationary"

    else:
        perc = df_critical["percentages"][index]
        return "The series have {}% chance of being stationary".format(100 - perc)
