from scipy.optimize import minimize
import numpy as np
import math
import matplotlib.pyplot as plt
from sge.utilities.evaluations import *


def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)
    return sig

def str_to_expr(text):
    text = text.replace("|_div_|", "/")
    text = text.replace("_exp_", "np.exp")
    text = text.replace("_sig_", "sigmoid")
    text = text.replace("x[0]", "t")
    return text

def optIndividual(x_array,y_array,individual,old_c):
    t=x_array
    fun = lambda cte : eval(str_to_expr(individual),{"cte":cte,"t":x_array,"sigmoid":sigmoid,"np":np})
    funobj= lambda cte :root_mean_squared_error(y_array,fun(cte))
    res = minimize(funobj,old_c, method='SLSQP',jac=False)

    new_c = res['x']

    new_output = fun(new_c)

    RMSE=res.fun
    return new_c,new_output,RMSE

