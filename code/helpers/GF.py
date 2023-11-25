# The code for evaluating the GF score of a model.
# Can find the evidence threshold which gives the best score.
YBARtrain = 0.05144011945525525



import bisect
import numpy as np
def printRankedCoefs(sklearnmodel):
    'Pulls the feature names and coefficients from a model and prints them in order of coeficient size.'
    coefficients = list(sklearnmodel.coef_)
    if len(coefficients) == 1: #classifier coef are 2D?
        coefficients = list(sklearnmodel.coef_)[0]
    coeflist = zip(coefficients, list(sklearnmodel.feature_names_in_))
    for coef, name in sorted(coeflist):
        print(name,coef)










# %% Define balanced accuracy but for regressors instead of classfiers. 

from sklearn.metrics import make_scorer
def simpleGF(y_true, y_pred, THRESHOLD = YBARtrain):
    '''This function makes 'balanced' predictions and then calculates GF.
    It just cuts off a threshold at the prior for U = 0.05144011945525525
    First line `y_true = y_true['y_tp1']` handles types in MEBDI22 specifically.'''
    y_true = y_true['y_tp1']
    y_pred = y_pred.ravel()
    tr0pr0 = np.sum([(y_true == 0)&(y_pred < THRESHOLD)])
    tr1pr1 = np.sum([(y_true == 1)&(y_pred >= THRESHOLD)])
    tr0 = np.sum([y_true == 0])
    tr1 = np.sum([y_true == 1])
    return (tr0pr0/tr0 + tr1pr1/tr1)/2 

def simpleGFU(y_true, y_pred, THRESHOLD = YBARtrain):
    '''This function makes 'balanced' predictions and then calculates GF.
    It just cuts off a threshold at the prior for U = 0.05144011945525525
    First line `y_true = y_true['y_tp1']` handles types in MEBDI22 specifically.'''
    y_true = y_true['y_tp1']
    y_pred = y_pred.ravel()
    tr0pr0 = np.sum([(y_true == 0)&(y_pred < THRESHOLD)])
    tr1pr1 = np.sum([(y_true == 1)&(y_pred >= THRESHOLD)])
    tr0 = np.sum([y_true == 0])
    tr1 = np.sum([y_true == 1])
    return tr1pr1/tr1


def trueGFandGFU(y_true, y_pred,):
    '''Similar to the above, but without a builtin threshold.
    This is essentially the metric used in the competition.
    For classification problems, it doesn't matter which is used.'''
    y_true = y_true['y_tp1']
    tr0 = np.sum([y_true == 0])
    tr1 = np.sum([y_true == 1])
    thisthreshold = tr1/(tr1+tr0) #ybar but for this specific input.
    tr0pr0 = np.sum([(y_true == 0)&(y_pred < thisthreshold)])
    tr1pr1 = np.sum([(y_true == 1)&(y_pred >= thisthreshold)])
    gfscore = (tr0pr0/tr0 + tr1pr1/tr1)/2 
    gfuscore = tr1pr1/tr1
    return gfscore, gfuscore


simpleGFscorer = make_scorer(simpleGF, greater_is_better=True)
simpleGFUscorer = make_scorer(simpleGFU, greater_is_better=True)





def thresholdCM(y_true, y_pred, THRESHOLD = YBARtrain):
    '''This function gives a confusion matrix. Counts for each category.
    It assumes series 1 is binary and series 2 is numerical.'''
    y_true = y_true['y_tp1']
    tr0pr0 = np.sum([(y_true == 0)&(y_pred < THRESHOLD)])
    tr0pr1 = np.sum([(y_true == 0)&(y_pred >= THRESHOLD)])
    tr1pr0 = np.sum([(y_true == 1)&(y_pred < THRESHOLD)])
    tr1pr1 = len(y_true) - (tr0pr0+tr0pr1+tr1pr0)
    #tr1pr1 = np.sum([(y_true == 1)&(y_pred >= THRESHOLD)])
    return tr0pr0,tr0pr1,tr1pr0,tr1pr1

def simpleCM(y_true, y_pred, THRESHOLD = YBARtrain):
    '''This function gives a confusion matrix. Counts for each category.
    It assumes both series are binary.'''
    y_true = y_true['y_tp1']
    tr0pr0 = np.sum([(y_true == 0)&(y_pred < THRESHOLD)])
    tr0pr1 = np.sum([(y_true == 0)&(y_pred >= THRESHOLD)])
    tr1pr0 = np.sum([(y_true == 1)&(y_pred < THRESHOLD)])
    tr1pr1 = len(y_true) - (tr0pr0+tr0pr1+tr1pr0)
    #tr1pr1 = np.sum([(y_true == 1)&(y_pred >= THRESHOLD)])
    return tr0pr0,tr0pr1,tr1pr0,tr1pr1







# %%

