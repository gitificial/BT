
import numpy as np
import pandas as pd

from decimal import *

test_scores = [[0.3, 0.8, 0.85, 0.7, 0.55, 0.75, 0.7, 0.7, 0.75, 0.6]]

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


