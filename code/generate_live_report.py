'''
This file is a modified version of the generate_report.py file,
intended for use in the live test run of the MEBDI 2022 ML competition.
Changes from the generate_report.py file are marked with a comment # LIVE RUN
'''


# %% Import data and modules
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import helpers.prepipelineprocessing as ppp
import helpers.GF as GF


training_dta = pd.io.stata.read_stata('../training_data/training_sample_2021_12_02_clean.dta')
raw_X = training_dta.drop(columns=['y_tp1']) # just to check for data leakage
y = training_dta[['y_tp1']]

# LIVE RUN - Load testing data
testing_dta = pd.io.stata.read_stata('../testing_data/test_sample_2021_12_02_clean.dta')
raw_X_test = testing_dta.drop(columns=['y_tp1'])
y_test = testing_dta[['y_tp1']]

#%% sklearn pipeline
NUM_ATTRIBS = [
    'earnweek',
    'earnhour_est',
    'logearnweek',
    'uhrsworkt','ahrsworkt',
    'ageint', # redundant ???
    'period_unrate','byage_unrate', # might risk overfitting
    'education_ordinal',  # redundantish with EDUCN_CATS
    'nchildren_ordinal', #redundantish with ppp.KIDS_CATS
    ]
CAT_ATTRIBS = [
    "empstat","statefip",
    "relate","marst","famsize","sex",
    "nchild",
    "nativity","hispan",
    "labforce","occ1990","ind1990","classwkr",
    "wkstat","paidhour", 
    'educ',  #redundantish. If I include this, then set start index of EDUCN to 1, end to -2
    "month", "year", #"yearmonth",# "yearmonth", "month", "year",
    "agebin", #'age', #'age' means one dummy per year; 'agebin' means only a few categories 
    #"race_orig", #redundantish, and causes problems
    ]
PPP_ATTRIBS = ppp.RACE_BINS + ppp.EDUCN_CATS[1:-2] + ppp.INDUSTRY_GROUPS + ppp.OCC_GROUPS + ppp.COMBOFEATURES + ppp.KIDS_CATS[1:-1]

# Fill in missing values and standardize scale of numeric variables.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
SimpleImputer.get_feature_names_out=(lambda self, names=None: self.feature_names_in_) #Reason: https://stackoverflow.com/a/69693003
num_pipeline = Pipeline([
        ('median_imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
        ('std_scaler', StandardScaler(with_mean=False)), #`with_mean=False` to keep the zeros at zero; `True` to set mean to zero.
    ])
from sklearn.preprocessing import OneHotEncoder 
from sklearn.feature_selection import VarianceThreshold
cat_pipeline = Pipeline([
        ("oh_encoder", OneHotEncoder(sparse=False, drop='if_binary')), # drop='first' or 'if_binary' or None ;handle_unknown='error'
        ("variance_thresh", VarianceThreshold(threshold=0.001)), # removes cats with fewer than ~300 entries
        #('std_scaler', StandardScaler(with_mean=False)), # Need to think about this. See here: https://stats.stackexchange.com/questions/359015/ridge-lasso-standardization-of-dummy-indicators
    ])
ppp_pipeline = Pipeline([
        ("variance_thresh", VarianceThreshold(threshold=0.001)), # this removes about 18 features
        #('std_scaler', StandardScaler(with_mean=False)),
    ])
from sklearn.compose import ColumnTransformer
full_pipeline = ColumnTransformer([
        ("num", num_pipeline, NUM_ATTRIBS),
        ("cat", cat_pipeline, CAT_ATTRIBS), 
        #("pst",'passthrough',PPP_ATTRIBS)
        ("pppp",ppp_pipeline,PPP_ATTRIBS)
    ], verbose_feature_names_out=False)

# %% Prepare (Training) Data
train_data_precleaned =  ppp.prepipeline_clean_all(raw_X)#, yfortimeseries=y) # LIVE RUN - verify live smoothing works. Yep, same results either way.
train_data_prepared = pd.DataFrame(full_pipeline.fit_transform(train_data_precleaned),
    columns=full_pipeline.get_feature_names_out())
train_data_prepared = ppp.postpipeline_drop_redundancies(train_data_prepared)
#train_data_prepared = train_data_prepared.copy() #.copy() defragments or something like that.
X = train_data_prepared.copy()



# %% LIVE RUN - Prepare Testing Data (Call transform instead of fit_transform in sklearn pipelines)
test_data_precleaned =  ppp.prepipeline_clean_all(raw_X_test)
test_data_prepared = pd.DataFrame(full_pipeline.transform(test_data_precleaned),
    columns=full_pipeline.get_feature_names_out())
test_data_prepared = ppp.postpipeline_drop_redundancies(test_data_prepared)
X_test = test_data_prepared.copy()







# %% Define simple voting ensemble
from helpers.GF import YBARtrain
def get_voted_predictions(pred1,pred2,pred3, threshold=YBARtrain):
    binary_pred1 = (pred1 > threshold).astype('int').ravel()
    binary_pred2 = (pred2 > threshold).astype('int').ravel()
    binary_pred3 = (pred3 > threshold).astype('int').ravel()
    print(binary_pred1)
    print(binary_pred2)
    print(binary_pred3)
    sum_predictions = binary_pred1 + binary_pred2 + binary_pred3
    voted_predictions = (sum_predictions >= 2).astype('int')
    return voted_predictions






# %%  Create Model objects
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=11,min_samples_split=0.03, splitter='random',class_weight='balanced',random_state=42)

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=7.5e-5,random_state=42)

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1000,random_state=42)





#%% Get scores from a cross validations
from helpers.GF import simpleGFscorer,simpleGFUscorer, simpleGF, simpleGFU


# LIVE RUN - Cross Validation Unnecessary for Live Run
'''
CVscores = {
    'dtree': {
        'GF': [],
        'GFU': [],
    },
    'lasso': {
        'GF': [],
        'GFU': [],
    },
    'ridge': {
        'GF': [],
        'GFU': [],
    },
    'voted': {
        'GF': [],
        'GFU': [],
    },
}

#  pred_lass = lasso.predict(fold_test_X)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for trainindices, testindices in skf.split(X,y):
    print('Fold')
    fold_train_X, fold_train_y = X.iloc[trainindices], y.iloc[trainindices]
    fold_test_X,  fold_test_y  = X.iloc[testindices],  y.iloc[testindices]
    
    dtree.fit(fold_train_X,fold_train_y)
    dtree_pred = dtree.predict(fold_test_X)
    CVscores['dtree']['GF'].append(simpleGF(fold_test_y,dtree_pred))
    CVscores['dtree']['GFU'].append(simpleGFU(fold_test_y,dtree_pred))
    #CVscores['dtree']['GF'].append(simpleGFscorer(dtree,fold_test_X,fold_test_y))
    #CVscores['dtree']['GFU'].append(simpleGFUscorer(dtree,fold_test_X,fold_test_y))
    
    lasso.fit(fold_train_X,fold_train_y)
    lasso_pred = lasso.predict(fold_test_X)
    CVscores['lasso']['GF'].append(simpleGF(fold_test_y,lasso_pred))
    CVscores['lasso']['GFU'].append(simpleGFU(fold_test_y,lasso_pred))
    
    ridge.fit(fold_train_X,fold_train_y)
    ridge_pred = ridge.predict(fold_test_X)
    CVscores['ridge']['GF'].append(simpleGF(fold_test_y,ridge_pred))
    CVscores['ridge']['GFU'].append(simpleGFU(fold_test_y,ridge_pred))

    voted_pred = get_voted_predictions(dtree_pred,lasso_pred,ridge_pred)
    CVscores['voted']['GF'].append(simpleGF(fold_test_y,voted_pred))
    CVscores['voted']['GFU'].append(simpleGFU(fold_test_y,voted_pred))
'''


#%% Displays scores

resultslist = []
def log(result):
    print(result)
    resultslist.append(result)


# LIVE RUN - Cross Validation Unnecessary for Live Run
'''
for model, v in CVscores.items():
    for metric, scorelist in v.items():
        scores = [round(x,5) for x in scorelist]
        average = round(np.mean(scorelist),5)
        log(f'{model} CV {metric}: {scores}, mean: {average}')
'''

#%% Refit the data on the entire training set, and save to data frame.
dtree.fit(X,y)
lasso.fit(X,y)
ridge.fit(X,y)

# LIVE RUN - change predictions to be based on testing data
dtree_predictions = dtree.predict(X_test)
lasso_predictions = lasso.predict(X_test)
ridge_predictions = ridge.predict(X_test)
voted_predictions = get_voted_predictions(dtree_predictions,lasso_predictions,ridge_predictions)

#%% LIVE RUN - change output predictions to be based on testing data
outputdf = y_test.copy()
outputdf['cpsid'] = testing_dta['cpsid'] # LIVE RUN - Include CPSID as a second identification variable
outputdf['dtree_pred'] = dtree_predictions
outputdf['lasso_pred'] = lasso_predictions
outputdf['ridge_pred'] = ridge_predictions
outputdf['voted_pred'] = voted_predictions



#%% LIVE RUN - Print GF scores to console.
testscores = {
    'dtree': {
        'GF': simpleGF(y_test, dtree_predictions),
        'GFU': simpleGFU(y_test, dtree_predictions),
    },
    'lasso': {
        'GF': simpleGF(y_test, lasso_predictions),
        'GFU': simpleGFU(y_test, lasso_predictions),
    },
    'ridge': {
        'GF': simpleGF(y_test, ridge_predictions),
        'GFU': simpleGFU(y_test, ridge_predictions),
    },
    'voted': {
        'GF': simpleGF(y_test, voted_predictions),
        'GFU': simpleGFU(y_test, voted_predictions),
    },
}
for model, v in testscores.items():
    for metric, score in v.items():
        log(f'{model} {metric}: {round(score,5)}')








# %% Output results

outputlocation = os.path.join(os.getcwd(),'results','testing_predictions_live.dta')# LIVE RUN - filename change
outputdf.to_stata(outputlocation)

reportlocation =  os.path.join(os.getcwd(),'results','metrics_log_live.txt')# LIVE RUN - filename change
with open(reportlocation,'w') as f:
    for line in resultslist:
        f.write(repr(line))
        f.write('\n')
    f.write('\n')
    # f.write(str(CVscores)) # LIVE RUN - Cross Validation Unnecessary for Live Run


# %% Print some interesting metrics
