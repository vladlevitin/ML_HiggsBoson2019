# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""
import numpy as np


def fill_nan(x, nan_value=np.nan, method='mean', filler=None):
    check_nans = (x == nan_value)
    if method=='mean':
        sum_cols = np.sum(x*(1-check_nans), axis=0, keepdims=True)
        sum_non_nans = np.sum(1-check_nans, axis=0, keepdims=True)
        mean_cols = sum_cols/sum_non_nans
        x = check_nans*mean_cols + (1-check_nans)*x
        filler = mean_cols
    elif method=='median':
        x = np.nan*check_nans + x*(1-check_nans)
        median_cols = np.nanmedian(x, axis=0)
        x = check_nans*median_cols + (1-check_nans)*x
        filler = median_cols
    elif method=='use_filler':
        x = check_nans*filler + (1-check_nans)*x
    else:
        raise NoSuchFillMethodException
    return x, filler

def interaction_terms(inp):
    x = inp.copy()
    res = np.zeros((x.shape[0],1))
    int_list = []
    for i in range(x.shape[1]):
        for j in range(i+1,x.shape[1]):
            a = x[:,i]*x[:,j]
            res = np.hstack([res,a.reshape(-1,1)])
            int_list.append((i,j))
    return res[:,1:], int_list

def var_cap(x, int_list):
    var = np.var(x, axis=0)
    top = np.argsort(abs(var))
    return top, np.array(int_list)[top]

def generate_interactions(inp, int_list):
    x = inp.copy()
    res = np.zeros((x.shape[0],1))
    for i in int_list:
        a = x[:,i[0]]*x[:,i[1]]
        res = np.hstack([res,a.reshape(-1,1)])
    return res[:,1:]

def poly_features(x, degree):
    """polynomial basis functions for input data x, for j=2 up to j=degree."""
    if len(x.shape)==1:
        x = x.reshape(-1,1)
    
    if degree==1:
        return x
    
    curr_deg = np.copy(x)
    x_poly = np.copy(curr_deg)
    
    for i in range(1,degree):
        curr_deg = np.multiply(curr_deg,x)
        x_poly = np.hstack((x_poly, curr_deg))
    
    return x_poly

def add_ones(x):
    return np.hstack([np.ones((len(x),1)),x])

def remove_outliers(x, outlier_thresh=None, conf_int=0.05):
    if not outlier_thresh:
        outlier_thresh = {
            'lower': (np.quantile(x, conf_int/2,axis =0 , keepdims=True)), 
            'upper': (np.quantile(x, 1-conf_int/2,axis =0 , keepdims=True))
        }

    mask_lower = x > outlier_thresh['lower']
    mask_upper = x < outlier_thresh['upper']
    x = (1-mask_upper) * (1-mask_lower)* x + mask_upper * outlier_thresh['upper'] + mask_lower * outlier_thresh['lower']

    return x,outlier_thresh



def standardize(x, norm_stats=None):
    if not norm_stats: 
        # Calculate the mean of the data    
        mean_cols = np.mean(x, axis=0, keepdims=True)
        # Calculate Standard Deviation of the data
        std_cols = np.std(x, axis=0, keepdims=True)
        # Save them in a dictionary 
        norm_stats = {'mean':mean_cols, 'std': std_cols}
    x = (x - norm_stats['mean'])/norm_stats['std']
    return x, norm_stats
