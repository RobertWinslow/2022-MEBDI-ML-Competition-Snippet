
import numpy as np
import pandas as pd



PRECALCULATED_TRAINING_URATE_MAP = {
    '2008January': 5.73285, '2008February': 5.78382, '2008March': 5.80967, '2008April': 5.83197, '2008May': 5.93333, '2008June': 5.95595, '2008July': 6.01736, '2008August': 6.04828, '2008September': 6.04938, '2008October': 6.13353, '2008November': 6.17365, '2008December': 6.20875, 
    '2009January': 6.20914, '2009February': 6.22535, '2009March': 6.20036, '2009April': 6.21926, '2009May': 6.27713, '2009June': 6.3204, '2009July': 6.35937, '2009August': 6.32558, '2009September': 6.27733, '2009October': 6.17277, '2009November': 6.13728, '2009December': 6.06131, 
    '2010January': 6.02248, '2010February': 5.97795, '2010March': 5.90013, '2010April': 5.86559, '2010May': 5.81263, '2010June': 5.77562, '2010July': 5.78958, '2010August': 5.72532, '2010September': 5.66814, '2010October': 5.55593, '2010November': 5.51199, '2010December': 5.47202, 
    '2011January': 5.43813, '2011February': 5.41045, '2011March': 5.39279, '2011April': 5.37751, '2011May': 5.37207, '2011June': 5.33295, '2011July': 5.33482, '2011August': 5.29833, '2011September': 5.24261, '2011October': 5.14384, '2011November': 5.15096, '2011December': 5.0777,
    '2012January': 5.08907, '2012February': 5.03636, '2012March': 4.98046, '2012April': 4.90617, '2012May': 4.80437, '2012June': 4.80842, '2012July': 4.82576, '2012August': 4.76792, '2012September': 4.64245, '2012October': 4.531, '2012November': 4.45755, '2012December': 4.36399, 
    '2013January': 4.32948, '2013February': 4.27093, '2013March': 4.22867, '2013April': 4.09449, '2013May': 4.02791, '2013June': 4.01096, '2013July': 3.95114, '2013August': 3.89524, '2013September': 3.83776, '2013October': 3.7884, '2013November': 3.70447, '2013December': 3.63268, 
    '2014January': 3.57149, '2014February': 3.57188, '2014March': 3.55383, '2014April': 3.50117, '2014May': 3.48554, '2014June': 3.45095, '2014July': 3.45357, '2014August': 3.44205, '2014September': 3.47171, '2014October': 3.44859, '2014November': 3.36574, '2014December': 3.32979
} # Made using radius 8



PERIODS = ['2008January', '2008February', '2008March', '2008April', '2008May',
    '2008June', '2008July', '2008August', '2008September', '2008October',
    '2008November', '2008December', '2009January', '2009February',
    '2009March', '2009April', '2009May', '2009June', '2009July',
    '2009August', '2009September', '2009October', '2009November',
    '2009December', '2010January', '2010February', '2010March', '2010April',
    '2010May', '2010June', '2010July', '2010August', '2010September',
    '2010October', '2010November', '2010December', '2011January',
    '2011February', '2011March', '2011April', '2011May', '2011June',
    '2011July', '2011August', '2011September', '2011October',
    '2011November', '2011December', '2012January', '2012February',
    '2012March', '2012April', '2012May', '2012June', '2012July',
    '2012August', '2012September', '2012October', '2012November',
    '2012December', '2013January', '2013February', '2013March', '2013April',
    '2013May', '2013June', '2013July', '2013August', '2013September',
    '2013October', '2013November', '2013December', '2014January',
    '2014February', '2014March', '2014April', '2014May', '2014June',
    '2014July', '2014August', '2014September', '2014October',
    '2014November', '2014December']



def getUrate(X,y,period):
    period_y = y.loc[X['yearmonth']==period,'y_tp1']
    numNU,numU = period_y.value_counts()
    Urate = numU/(numNU+numU)
    return Urate

def mysmoother(inputlist,radius):
    'Endpoints get less smoothed this way, but effects are less extreme than convolving.'
    averages = []
    for i,v in enumerate(inputlist):
        windowleft = max(0,i-radius)
        windowright = i+1+radius
        window = inputlist[windowleft:windowright]
        averages.append(np.mean(window))
    return averages

def constructSmoothUrateDict(X,y, radius=8):
    '''This only works if X has a 'yearmonth' column and y has a 'y_tp1' column.'''
    Urates = [getUrate(X,y,period)*100 for period in PERIODS]
    UrateSmooth = mysmoother(Urates,radius)
    assert len(UrateSmooth) == len(PERIODS)
    uratedict = {}
    for period,rate in zip(PERIODS,UrateSmooth):
        uratedict[period] = round(rate,5)
    return uratedict