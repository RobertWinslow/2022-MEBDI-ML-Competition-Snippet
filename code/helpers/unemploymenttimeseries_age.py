
import numpy as np
import pandas as pd

#%%

PRECALCULATED_TRAINING_URATE_MAP = {
    20: 10.41161, '20': 10.41161, 21: 9.83588, '21': 9.83588, 22: 9.20892, '22': 9.20892, 23: 8.81242, '23': 8.81242, 24: 8.16162, '24': 8.16162, 25: 7.46747, '25': 7.46747, 
    26: 6.92888, '26': 6.92888, 27: 6.45369, '27': 6.45369, 28: 6.08764, '28': 6.08764, 29: 5.76573, '29': 5.76573, 
    30: 5.48625, '30': 5.48625, 31: 5.24792, '31': 5.24792, 32: 5.17639, '32': 5.17639, 33: 5.03419, '33': 5.03419, 34: 4.94701, '34': 4.94701, 35: 4.8487, '35': 4.8487, 
    36: 4.81946, '36': 4.81946, 37: 4.82974, '37': 4.82974, 38: 4.8255, '38': 4.8255, 39: 4.74671, '39': 4.74671, 
    40: 4.68856, '40': 4.68856, 41: 4.59086, '41': 4.59086, 42: 4.58705, '42': 4.58705, 43: 4.55469, '43': 4.55469, 44: 4.44915, '44': 4.44915, 45: 4.4724, '45': 4.4724, 
    46: 4.56102, '46': 4.56102, 47: 4.58136, '47': 4.58136, 48: 4.65129, '48': 4.65129, 49: 4.70074, '49': 4.70074, 
    50: 4.69342, '50': 4.69342, 51: 4.78882, '51': 4.78882, 52: 4.70936, '52': 4.70936, 53: 4.55409, '53': 4.55409, 54: 4.59421, '54': 4.59421, 55: 4.54112, '55': 4.54112, 
    56: 4.49309, '56': 4.49309, 57: 4.52914, '57': 4.52914, 58: 4.43527, '58': 4.43527, 59: 4.40317, '59': 4.40317, 
    60: 4.46189, '60': 4.46189, 61: 4.43461, '61': 4.43461, 62: 4.39773, '62': 4.39773, 63: 4.4704, '63': 4.4704, 64: 4.45305, '64': 4.45305
} # Made using radius 3



AGES = list(range(20,65))

def getUrate(X,y,age):
    age = str(age)
    if age == '64': # not enough people
        age ='63'
    period_y = y.loc[X['age']==age,'y_tp1']
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
        # make endpoints 'sticky'
        windowgap = (2*radius+1)-len(window)
        window = window + [v]*windowgap
        averages.append(np.mean(window))
    return averages

def constructSmoothUrateDict(X,y, radius=3):
    '''This only works if X has a 'age' column and y has a 'y_tp1' column.'''
    Urates = [getUrate(X,y,age)*100 for age in AGES]
    UrateSmooth = mysmoother(Urates,radius)
    assert len(UrateSmooth) == len(AGES)
    uratedict = {}
    for age,rate in zip(AGES,UrateSmooth):
        uratedict[age] = round(rate,5)
        uratedict[str(age)] = round(rate,5)
    return uratedict
# %%
