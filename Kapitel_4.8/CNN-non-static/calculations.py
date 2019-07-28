
import numpy as np
import pandas as pd

from decimal import *

test_scores = [[0.85, 0.6, 0.6, 0.9, 0.7, 0.9, 0.7, 0.9, 0.55, 0.75], [0.8, 0.9, 0.775, 0.9, 0.875, 0.85, 0.775, 0.725, 0.7, 0.925], [0.75, 0.76666665, 0.8666667, 0.81666666, 0.8, 0.8, 0.8666667, 0.8833333, 0.81666666, 0.8666667], [0.875, 0.8375, 0.75, 0.8125, 0.8625, 0.7625, 0.8375, 0.8875, 0.8, 0.9125], [0.87, 0.82, 0.76, 0.88, 0.87, 0.82, 0.88, 0.84, 0.83, 0.84], [0.78333336, 0.85833335, 0.81666666, 0.825, 0.8, 0.89166665, 0.8666667, 0.80833334, 0.8666667, 0.85], [0.8214286, 0.75714284, 0.8357143, 0.85, 0.7785714, 0.87857145, 0.8214286, 0.9, 0.8428571, 0.8142857], [0.84375, 0.84375, 0.8, 0.89375, 0.8375, 0.85625, 0.85625, 0.825, 0.86875, 0.9], [0.8611111, 0.87777776, 0.8388889, 0.78333336, 0.85, 0.84444445, 0.8388889, 0.84444445, 0.84444445, 0.82222223], [0.88, 0.79, 0.795, 0.845, 0.845, 0.835, 0.795, 0.845, 0.86, 0.84], [0.875, 0.8125, 0.85, 0.835, 0.85, 0.855, 0.81, 0.8425, 0.805, 0.855], [0.85, 0.87, 0.80833334, 0.815, 0.8516667, 0.85, 0.8466667, 0.865, 0.8433333, 0.82666665], [0.8575, 0.85375, 0.87375, 0.84375, 0.8225, 0.85125, 0.85, 0.84625, 0.85125, 0.86875]]

n = 10


test_scores_mean = np.mean(test_scores, axis=1)

test_scores_mean = list(np.around(np.array(test_scores_mean),3))

test_scores_std = np.std(test_scores, axis=1)
test_scores_std = list(np.around(np.array(test_scores_std),3))

test_scores_stdfailure = test_scores_std / np.sqrt(n)
test_scores_stdfailure = list(np.around(np.array(test_scores_stdfailure),3))
print(test_scores_stdfailure)

KIVneg = test_scores_mean - (np.asarray(1.96) * test_scores_stdfailure)
KIVneg = list(np.around(np.array(KIVneg),3))

KIVpos = test_scores_mean + (np.asarray(1.96) * test_scores_stdfailure)
KIVpos = list(np.around(np.array(KIVpos),3))

print("means: ", test_scores_mean)
print("std: ", test_scores_std)
print("stdfailure: ", test_scores_stdfailure)

print("KIVneg:", KIVneg)
print("KIVpos:", KIVpos)

df = pd.DataFrame({'mean':test_scores_mean, 'std':test_scores_std, 'stdfailure':test_scores_stdfailure, 'KIVneg':KIVneg, 'KIVpos':KIVpos})

print (df)

# df.to_csv('table.csv')


