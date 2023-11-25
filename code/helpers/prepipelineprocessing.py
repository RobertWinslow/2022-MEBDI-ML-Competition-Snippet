# PRE-SKLEARN CLEANING STEP
#%%
import numpy as np
import pandas as pd
from . import ind1990codes as ind
from . import occ1990codes as occ
from . import unemploymenttimeseries as uts
from . import unemploymenttimeseries_age as utsa

#%%

'''# Define weights for inflation adjustment
CPI_1999_DICT = {  # OUTSIDE DATA. Do not use in final model!
    2007: 0.826,
    2008: 0.804,
    2009: 0.774,
    2010: 0.777,
    2011: 0.764,
    2012: 0.741,
    2013: 0.726,
    2014: 0.715,
    2015: 0.704,
    2016: 0.703,
}  # OUTSIDE DATA. Do not use in final model!
# PER FEEDBACK FROM JUDGES, THE ALGORITHM SHOULDN'T USE OUTSIDE DATA, NOT EVEN FOR DEFLATION
'''
'''
PRECALCULATED_RAW_TRAINING_UNRATE_MAP = {
    '2008January':5.104,'2008February':5.262,'2008March':6.304,'2008April':5.869,'2008May':5.59,'2008June':5.889,'2008July':6.253,'2008August':5.84,'2008September':5.484,'2008October':6.243,'2008November':6.068,'2008December':6.077,
    '2009January':7.15,'2009February':6.25,'2009March':6.877,'2009April':6.512,'2009May':6.067,'2009June':6.534,'2009July':5.944,'2009August':6.901,'2009September':5.875,'2009October':5.866,'2009November':5.464,'2009December':6.574,
    '2010January':6.823,'2010February':6.22,'2010March':6.905,'2010April':5.494,'2010May':5.257,'2010June':5.372,'2010July':5.647,'2010August':5.586,'2010September':5.852,'2010October':5.31,'2010November':5.211,'2010December':5.357,
    '2011January':6.001,'2011February':5.246,'2011March':6.103,'2011April':4.372,'2011May':5.602,'2011June':4.916,'2011July':5.473,'2011August':6.225,'2011September':4.918,'2011October':4.786,'2011November':5.072,'2011December':5.387,
    '2012January':5.493,'2012February':5.187,'2012March':5.342,'2012April':4.591,'2012May':4.41,'2012June':4.322,'2012July':5.367,'2012August':4.858,'2012September':4.565,'2012October':4.706,'2012November':3.965,'2012December':4.21,
    '2013January':4.495,'2013February':4.986,'2013March':5.081,'2013April':4.089,'2013May':3.254,'2013June':3.598,'2013July':3.938,'2013August':3.751,'2013September':4.004,'2013October':3.414,'2013November':3.603,'2013December':3.086,
    '2014January':3.726,'2014February':4.277,'2014March':3.689,'2014April':3.015,'2014May':3.233,'2014June':3.656,'2014July':3.56,'2014August':3.861,'2014September':3.049,'2014October':3.261,'2014November':3.292,'2014December':3.043}
# Calculated from the training data.
# Not actually currently used.
# Done: Smooth the above out to prevent overfitting.
#    The unemploymenttimeseries module can now generate the thing or provide a pregenerated version.
'''

# Cleaning the Hours and Earnings variables
def prepipeline_clean_hourly(df):
    # AHRSWORKT
    # Change NIU to 0
    df.loc[df['ahrsworkt']==999,'ahrsworkt'] = 0
    # Cap outliers to 16 hours per day. 
    df.loc[df['ahrsworkt']>112,'ahrsworkt'] = 112

    # UHRSWORKT
    # change NIU to 0
    df.loc[df['uhrsworkt']=='NIU','uhrsworkt'] = 0
    # Remove 'Hours Vary' entries
    df.loc[df['uhrsworkt']=='Hours vary','uhrsworkt'] = np.nan
    # Cap outliers to 16 hours per day. 
    df.loc[df['uhrsworkt']>112,'uhrsworkt'] = 112

    # EARNWEEK
    # Remove NIU values
    df.loc[df['earnweek']==9999.99,'earnweek'] = np.nan
    # Impute zero earnings for those without work.
    df.loc[((df['empstat']!='Has job, not at work last week')&(df['empstat']!='At work')), 'earnweek'] = 0
    # Convert earnings to 1999 dollars
    #deflatorfunc = lambda row: row['earnweek']*CPI_1999_DICT[row['year']] # OUTSIDE DATA. Do not use in final model!
    #df['earnweek_real'] = df.apply(deflatorfunc,axis=1) # OUTSIDE DATA. Do not use in final model!

    # EARNHOUR_EST
    # Make crude estimate of hourly earnings.
    def earnhour_est_func(row):
        if row['earnweek'] == 0:
            return 0
        elif row['uhrsworkt'] > 0:
            return row['earnweek']/row['uhrsworkt']
        else:
            return np.nan
    df['earnhour_est'] = df.apply(earnhour_est_func,axis=1)
    # Deal with outliers.
    df.loc[df['earnhour_est']>200,'earnhour_est'] = 200
    # Adjust for inflation.
    #deflatorfunc_hourly = lambda row: row['earnhour_est']*CPI_1999_DICT[row['year']] # OUTSIDE DATA. Do not use in final model!
    #df['earnhour_real'] = df.apply(deflatorfunc_hourly,axis=1) # OUTSIDE DATA. Do not use in final model!

    # LOGEARNWEEK
    df['logearnweek'] = np.log(df['earnweek'])
    #df['logearnweek_real'] = np.log(df['earnweek_real']) # OUTSIDE DATA. Do not use in final model!
    # Remove bottom outliers.
    df.loc[df['logearnweek']<0,'logearnweek'] = 0
    #df.loc[df['logearnweek_real']<0,'logearnweek_real'] = 0 # OUTSIDE DATA. Do not use in final model!

# Simplify Racial Dummy Variables
RACE_BINS = ['racec_white', 'racec_black', 'racec_asian', 'racec_pacis', 'racec_amind', 'racec_mixed']
def prepipeline_clean_race(df):
    "This function unpacks the combinatoric race dummy variables into a smaller set of dummy variables."
    df['race_orig'] = df['race'] # original unadjusted race
    df = pd.get_dummies(df, columns = ['race'])
    race_dummies = ['race_White', 'race_Black/Negro', 'race_American Indian/Aleut/Eskimo', 'race_Asian only', 'race_Hawaiian/Pacific Islander only', 'race_White-Black', 'race_White-American Indian', 'race_White-Asian', 'race_White-Hawaiian/Pacific Islander', 'race_Black-American Indian', 'race_Black-Asian', 'race_Black-Hawaiian/Pacific Islander', 'race_American Indian-Asian', 'race_Asian-Hawaiian/Pacific Islander', 'race_White-Black-American Indian', 'race_White-Black-Asian', 'race_White-American Indian-Asian', 'race_White-Asian-Hawaiian/Pacific Islander', 'race_White-Black-American Indian-Asian', 'race_American Indian-Hawaiian/Pacific Islander', 'White-Black--Hawaiian/Pacific Islander', 'race_White-American Indian-Hawaiian/Pacific Islander', 'race_Black-American Indian-Asian', 'race_White-American Indian-Asian-Hawaiian/Pacific Islander', 'race_Two or three races, unspecified', 'race_Four or five races, unspecified',]
    for race_dummy in race_dummies:
        # This should not trigger if the test data was constructed correctly. I'm just being paranoid.
        if race_dummy not in list(df):
            print(f'Preprocessing: adding temporary dummy variable for {race_dummy}')
            df[race_dummy] = 0 # Used to be data[race_dummy] = 0; gawd I cannot believe I wasted an hour tracking down such a simple bug.
    # sum race categories; uses 'racec_' (race combined) as prefix to avoid name collisions.
    df['racec_white'] = df[['race_White','race_White-Black','race_White-American Indian','race_White-Asian','race_White-Hawaiian/Pacific Islander','race_White-Black-American Indian','race_White-Black-Asian','race_White-American Indian-Asian','race_White-Asian-Hawaiian/Pacific Islander','race_White-Black-American Indian-Asian','race_White-American Indian-Hawaiian/Pacific Islander','race_White-American Indian-Asian-Hawaiian/Pacific Islander', 'White-Black--Hawaiian/Pacific Islander']].sum(axis=1)
    df['racec_black'] = df[['race_Black/Negro','race_White-Black','race_Black-American Indian','race_Black-Asian','race_Black-Hawaiian/Pacific Islander','race_White-Black-American Indian','race_White-Black-Asian','race_White-Black-American Indian-Asian','race_Black-American Indian-Asian', 'White-Black--Hawaiian/Pacific Islander']].sum(axis=1)
    df['racec_amind'] = df[['race_American Indian/Aleut/Eskimo','race_White-American Indian','race_Black-American Indian','race_American Indian-Asian','race_White-Black-American Indian','race_White-American Indian-Asian','race_White-Black-American Indian-Asian','race_American Indian-Hawaiian/Pacific Islander','race_White-American Indian-Hawaiian/Pacific Islander','race_Black-American Indian-Asian','race_White-American Indian-Asian-Hawaiian/Pacific Islander']].sum(axis=1)
    df['racec_asian'] = df[['race_Asian only','race_White-Asian','race_Black-Asian','race_American Indian-Asian','race_Asian-Hawaiian/Pacific Islander','race_White-Black-Asian','race_White-American Indian-Asian','race_White-Asian-Hawaiian/Pacific Islander','race_White-Black-American Indian-Asian','race_Black-American Indian-Asian','race_White-American Indian-Asian-Hawaiian/Pacific Islander']].sum(axis=1)
    df['racec_pacis'] = df[['race_Hawaiian/Pacific Islander only','race_White-Hawaiian/Pacific Islander','race_Black-Hawaiian/Pacific Islander','race_Asian-Hawaiian/Pacific Islander','race_White-Asian-Hawaiian/Pacific Islander','race_American Indian-Hawaiian/Pacific Islander','race_White-American Indian-Hawaiian/Pacific Islander','race_White-American Indian-Asian-Hawaiian/Pacific Islander', 'White-Black--Hawaiian/Pacific Islander']].sum(axis=1)
    df['racec_mixed'] = df[['race_White-Black','race_White-American Indian','race_White-Asian','race_White-Hawaiian/Pacific Islander','race_Black-American Indian','race_Black-Asian','race_Black-Hawaiian/Pacific Islander','race_American Indian-Asian','race_Asian-Hawaiian/Pacific Islander','race_White-Black-American Indian','race_White-Black-Asian','race_White-American Indian-Asian','race_White-Asian-Hawaiian/Pacific Islander','race_White-Black-American Indian-Asian','race_American Indian-Hawaiian/Pacific Islander','race_White-American Indian-Hawaiian/Pacific Islander','race_Black-American Indian-Asian','race_White-American Indian-Asian-Hawaiian/Pacific Islander','race_Two or three races, unspecified','race_Four or five races, unspecified']].sum(axis=1)
    df = df.drop(columns=race_dummies)
    return df

# Turn education into a mostly hierarchical set of dummies.
EDUCN_CATS = ["educn_4","educn_6","educn_8","educn_9","educn_10","educn_11","educn_12","educn_hs","educn_somcol","educn_assoct","educn_bachlr","educn_master","educn_profnl","educn_doctor"]
def prepipeline_clean_educ(df):
    '''The **educ** feature is partially ordinal. High degrees imply low degrees.
    so I'm setting up the dummy variables with overlap to reflect this.
    This allows a decision tree to e.g. compare (HS grads VS non-HS grads) instead of (HS grads who didn't go to college VS everyone else)'''
    #start by mapping education levels to numbers
    # I don't use the resulting ordinal variable, but who knows, maybe I will decide to later.
    educOrdinalMap = {"None or preschool":0, "Grades 1, 2, 3, or 4":4, 
        "Grades 5 or 6":6, "Grades 7 or 8":8, "Grade 9":9, "Grade 10":10, "Grade 11":11, "12th grade, no diploma":12,
        "High school diploma or equivalent":12.5, "Some college but no degree":13, 
        "Associate's degree, occupational/vocational program":14, "Associate's degree, academic program":14,
        "Bachelor's degree":16, "Master's degree":18, "Professional school degree":21, "Doctorate degree":21}
    df['education_ordinal'] = df['educ'].map(educOrdinalMap)
    # Now create a sequence of nested binary variables.
    # prefix is educn_ (education nested) instead of educ_ to avoid name collisions
    df['educn_4'] = (df['education_ordinal'] >= 4).astype('int')
    df['educn_6'] = (df['education_ordinal'] >= 6).astype('int')
    df['educn_8'] = (df['education_ordinal'] >= 8).astype('int')
    df['educn_9'] = (df['education_ordinal'] >= 9).astype('int')
    df['educn_10'] = (df['education_ordinal'] >= 10).astype('int')
    df['educn_11'] = (df['education_ordinal'] >= 11).astype('int')
    df['educn_12'] = (df['education_ordinal'] >= 12).astype('int')
    df['educn_hs'] = (df['education_ordinal'] >= 12.5).astype('int')
    df['educn_somcol'] = (df['education_ordinal'] >= 13).astype('int')
    df['educn_assoct'] = (df['education_ordinal'] >= 14).astype('int')
    df['educn_bachlr'] = (df['education_ordinal'] >= 16).astype('int')
    df['educn_master'] = (df['education_ordinal'] >= 18).astype('int')
    df['educn_profnl'] = (df['educ'] == "Professional school degree").astype('int')
    df['educn_doctor'] = (df['educ'] == "Doctorate degree").astype('int') #short for doctorate. Medical practitioners often have profnl degree.
    return df


# Turn education into a mostly hierarchical set of dummies.
KIDS_CATS = ['kids_geq1','kids_geq2','kids_geq3','kids_geq4','kids_geq5','kids_geq6','kids_geq7','kids_geq8','kids_geq9',] #kids_nope is redundant
def prepipeline_clean_kids(df):
    '''Created nested dummies for number of children'''
    #start by mapping nchild categories to ordinals
    nchildOrdinalMap = {'0 children present': 0, '1 child present': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9+': 9}
    df['nchildren_ordinal'] = df['nchild'].map(nchildOrdinalMap)
    # Now create a sequence of nested binary variables.
    # prefix is educn_ (education nested) instead of educ_ to avoid name collisions
    df['kids_geq1'] = (df['nchildren_ordinal'] >= 1).astype('int')
    df['kids_geq2'] = (df['nchildren_ordinal'] >= 2).astype('int')
    df['kids_geq3'] = (df['nchildren_ordinal'] >= 3).astype('int')
    df['kids_geq4'] = (df['nchildren_ordinal'] >= 4).astype('int')
    df['kids_geq5'] = (df['nchildren_ordinal'] >= 5).astype('int')
    df['kids_geq6'] = (df['nchildren_ordinal'] >= 6).astype('int')
    df['kids_geq7'] = (df['nchildren_ordinal'] >= 7).astype('int')
    df['kids_geq8'] = (df['nchildren_ordinal'] >= 8).astype('int')
    df['kids_geq9'] = (df['nchildren_ordinal'] >= 9).astype('int')
    return df



# Combine industries with hierarchical grouping
INDUSTRY_GROUPS = ['indg_'+ group for group in  ind.INDUSTRYGROUPS.keys()]
def prepipeline_clean_industry(df):
    '''The **ind1990** feature has categories which are very sparse.
    But they are hierarchical in a sensible way. This function groups industries together.'''
    #existingcols = list(df)
    df[INDUSTRY_GROUPS] = [0]*len(INDUSTRY_GROUPS)
    for industry, groups in ind.INDUSTRYGROUPMAP.items():
        for group in groups:
            groupcol = 'indg_'+ group
            df.loc[df['ind1990']==industry, groupcol] = 1 # mark the rows of the relevant column
    return df

# same for occupations
OCC_GROUPS = ['occg_'+ group for group in  occ.OCCGROUPS.keys()]
def prepipeline_clean_occupation(df):
    '''The **occ1990** feature has categories which are very spares.
    But they are hierarchical in a sensible way. This function groups occupations together.'''
    # Create columns for each new dummy.
    # Some of these overlap due to nested structure
    df[OCC_GROUPS] = [0]*len(OCC_GROUPS)
    for occupation, groups in occ.OCCGROUPMAP.items():
        for group in groups:
            groupcol = 'occg_'+ group
            df.loc[df['occ1990']==occupation, groupcol] = 1 # mark the rows of the relevant column
    return df

# Process/Simplify Age Variable. 
AGEBINS = [0,25,30,35,40,45,50,55,60,np.inf] # first should be <20 (ie 0) because of pd.cut quirks; last should be 65+
AGEBIN_NAMES = [f'{ageLow+1}-{ageHigh}' for ageLow, ageHigh in zip(AGEBINS[:],AGEBINS[1:])] 
def prepipeline_clean_age(df,yfortimeseries):
    '''This function does the per-sklearn processing on the age parameter.
    There are three possibilities for handling age:
        CATEGORY involves mapping each age to its own variable, but this will be handled later, inside the sklearn pipeline. 
        BIN maps ages to a smaller set of categories, which will then be one-hot-encoded inside the pipeline.
        NUMERICAL requires us to convert the ages to integers, which will later be scaled inside the pipeline
    There's no reason not to create all three columns, now that I think about it. sklearn needs to select them later.'''
    df['ageint'] = df['age'].astype('int')
    df['agebin'] = pd.cut(df['ageint'], AGEBINS, labels=AGEBIN_NAMES)
    # Now create a smoothed time series (age series?) of unemployment rates
    if isinstance(yfortimeseries, pd.DataFrame):
        print("Calculating smoothed unemployment rate by age.")
        ageseriesmap = utsa.constructSmoothUrateDict(df,yfortimeseries)
    else:
        ageseriesmap = utsa.PRECALCULATED_TRAINING_URATE_MAP
    df['byage_unrate'] = df['age'].map(ageseriesmap)
    return df

# Conjoin year and month into a single category, use that to create new variables
def prepipeline_clean_timeseries(df,yfortimeseries):
    '''This function creates a combined categorical series for year and month.
    Later, sklearn's pipeline can then create dummies for each individual time period.'''
    def combine_year_and_month(row):
        return str(row['year']) + row['month']
    df['yearmonth'] = df.apply(combine_year_and_month,axis=1).astype("category")
    # As it stands, the category is sorted alphabetically, I want to sort it chronologically
    yearsnippets = [str(y) for y in sorted(df['year'].unique())]
    monthsnippets = df['month'].cat.categories
    yearmonthlist = [y+m for y in yearsnippets for m in monthsnippets]
    df['yearmonth'] = df['yearmonth'].cat.set_categories(yearmonthlist)
    # Now create a smoothed time series of unemployment rates
    if isinstance(yfortimeseries, pd.DataFrame):
        print("Calculating smoothed unemployment rate by period.")
        timeseriesmap = uts.constructSmoothUrateDict(df,yfortimeseries)
    else:
        timeseriesmap = uts.PRECALCULATED_TRAINING_URATE_MAP
    df['period_unrate'] = df['yearmonth'].map(timeseriesmap)
    return df

# Specific 'most informative' combinations of empstat or wkstat chosen based on analysis in simple.py
# also, self employment, because that seems interesting.
from collections import defaultdict
COMBOFEATURES = ['empcombo','wkcombo','class_selfemp']
empcombomap = defaultdict(int)
for empstatcat in ['Unemployed, experienced worker', 'Unemployed, new worker', 'NILF, unable to work', 'NILF, other', 'NILF, retired']:
    empcombomap[empstatcat] = 1
wkcombomap = defaultdict(int)
for wkstatcat in ['Part-time hours, usually part-time for economic reasons', 'Part-time for economic reasons, usually full-time', 'Unemployed, seeking full-time work', 'NIU, blank, or not in labor force', 'Unemployed, seeking part-time work', 'Full-time hours, usually part-time for economic reasons', 'Not at work, usually part-time',]:
    wkcombomap[wkstatcat] = 1
classcombomap = defaultdict(int)
classcombomap['Self-employed, not incorporated'] = 1
classcombomap['Self-employed, incorporated'] = 1
def prepipeline_add_combos(df):
    df['empcombo'] = df['empstat'].map(empcombomap)
    df['wkcombo'] = df['wkstat'].map(wkcombomap)
    df['class_selfemp'] = df['classwkr'].map(classcombomap)
    return df


# Apply the above steps and also create dummy variables.
def prepipeline_clean_all(originaldata, yfortimeseries=None):
    '''The yfortimeseries variable should be left as None when applying to testing data.
    I left in the option of recalculating it because I want to do train test splits.
    '''
    df = originaldata.copy()
    prepipeline_clean_hourly(df)
    df = prepipeline_clean_race(df)
    df = prepipeline_clean_age(df,yfortimeseries)
    df = prepipeline_clean_timeseries(df,yfortimeseries)
    df = prepipeline_clean_educ(df)
    df = prepipeline_clean_kids(df)
    df = prepipeline_clean_industry(df)
    df = prepipeline_clean_occupation(df)
    df = prepipeline_add_combos(df)
    return df



def prepipeline_clean_basics(originaldata):
    '''The yfortimeseries variable should be left as None when applying to testing data.
    I left in the option of recalculating it because I want to do train test splits.
    '''
    df = originaldata.copy()
    prepipeline_clean_hourly(df)
    # delete the engineered features (keep logearnweek)
    df = df.drop(columns = ['earnhour_est'])
    #df = prepipeline_clean_race(df)
    # REMINDER: With this setup,
    # race needs to be included in the sklearn pipeline,
    # as does age, education, children, industry, and occupation
    return df




# %% REDUNDANT LABELS -----------------------------------------------
# There are a few sets of categories that are identical in the training data.


categories_isinmilitary = ['empstat_Armed Forces','labforce_NIU','classwkr_Armed forces']
categories_wasinmilitary = ['occ1990_Military', 'ind1990_Armed Forces, branch not specified']
# first set is current status, second asks about most recent job.

categories_neverworked = ['occ1990_NIU','classwkr_NIU','ind1990_NIU']
# Persons age 15+ who ever worked. (definitions not identical. But same set in training dta)

categories_laborforce = ['wkstat_NIU, blank, or not in labor force', 'labforce_Yes, in the labor force']
# These two are perfectly complementary, rather than being identical.

redundant_categories = categories_isinmilitary[1:] + categories_wasinmilitary[1:] +categories_neverworked[1:]+categories_laborforce[1:]
# My reasoning is empstat et al are a subset of the most informative combined metric I generated
# so occ1990_military will be more valuable. ???
#redundant_categories = list(set(redundant_categories)) # ensure uniqueness

redundant_categories += ['ind1990_'+cat for cat in ind.REDUNDANTINDUSTRIES]
redundant_categories += ['occ1990_'+cat for cat in occ.REDUNDANTOCCUPATIONS]+['occg_MILITARY OCCUPATIONS']

def postpipeline_drop_redundancies(df):
    '''This function drops categories which are duplicated in the training data.
    By duplicated, I mean there are two dummies which have the same value for every row.
    This function should be run after sklearn's pipeline.'''
    cats_to_drop = []
    dfcols = list(df)
    for cat in redundant_categories:
        if cat in dfcols:
            cats_to_drop.append(cat)
    cats_to_drop = list(set(cats_to_drop)) # ensure uniqueness
    df = df.drop(columns = cats_to_drop)
    return df
