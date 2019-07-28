
import numpy as np
import pandas as pd

from decimal import *

test_scores = [[0.699999988079071, 0.550000011920929, 0.550000011920929, 0.550000011920929, 0.4000000059604645, 0.6499999761581421, 0.800000011920929, 0.6000000238418579, 0.75, 0.75], [0.574999988079071, 0.42500001192092896, 0.44999998807907104, 0.550000011920929, 0.675000011920929, 0.6499999761581421, 0.6499999761581421, 0.75, 0.5249999761581421, 0.5], [0.699999988079071, 0.4833333194255829, 0.4833333194255829, 0.5666666626930237, 0.6666666865348816, 0.5833333134651184, 0.5833333134651184, 0.6000000238418579, 0.4833333194255829, 0.6000000238418579], [0.4125, 0.6625, 0.5625, 0.6625, 0.6625, 0.6875, 0.625, 0.6875, 0.7875, 0.625], [0.7299999952316284, 0.72, 0.6799999952316285, 0.6599999904632569, 0.7000000071525574, 0.7399999904632568, 0.7, 0.7199999904632568, 0.76, 0.7100000095367431], [0.7666666587193807, 0.725000011920929, 0.7833333293596904, 0.7749999920527141, 0.7416666746139526, 0.6333333253860474, 0.7249999880790711, 0.6500000119209289, 0.7, 0.6833333253860474], [0.6857142865657806, 0.7642857159887041, 0.7428571428571429, 0.8, 0.7785714302744184, 0.7428571445601327, 0.6928571445601327, 0.7500000017029899, 0.7214285714285714, 0.7571428554398673], [0.775, 0.625, 0.7625, 0.74375, 0.73125, 0.73125, 0.71875, 0.74375, 0.75, 0.78125], [0.7444444404708015, 0.7388888941870795, 0.816666669315762, 0.7055555635028415, 0.7222222142749363, 0.7833333280351427, 0.7722222182485793, 0.6833333412806193, 0.7555555595291985, 0.8000000066227383], [0.75, 0.725, 0.735, 0.82, 0.815, 0.815, 0.775, 0.775, 0.75, 0.82], [0.815, 0.8525, 0.8675, 0.8375, 0.8425, 0.835, 0.87, 0.8725, 0.805, 0.8575], [0.8299999992052715, 0.8766666666666667, 0.8649999992052714, 0.8466666674613953, 0.8666666674613953, 0.8500000007947286, 0.8533333325386048, 0.8449999992052714, 0.855, 0.865], [0.86375, 0.85625, 0.87375, 0.84875, 0.87375, 0.8525, 0.84875, 0.85875, 0.87375, 0.8575]]

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

df.to_csv('table.csv')

