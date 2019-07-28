
import numpy as np
import pandas as pd

from decimal import *

test_scores = [[0.65, 0.65, 0.6, 0.75, 0.55, 0.65, 0.6, 0.45, 0.8, 0.65], [0.575, 0.6, 0.575, 0.525, 0.55, 0.575, 0.575, 0.575, 0.625, 0.675], [0.5, 0.5, 0.716667, 0.666667, 0.45, 0.65, 0.43333299999999997, 0.533333, 0.55, 0.55], [0.5625, 0.675, 0.475, 0.5625, 0.5875, 0.6, 0.65, 0.6, 0.5, 0.575], [0.61, 0.59, 0.53, 0.55, 0.56, 0.57, 0.55, 0.47, 0.56, 0.44], [0.525, 0.575, 0.541667, 0.608333, 0.583333, 0.558333, 0.725, 0.725, 0.633333, 0.558333], [0.735714, 0.621429, 0.7071430000000001, 0.735714, 0.728571, 0.5857140000000001, 0.635714, 0.7071430000000001, 0.7, 0.778571], [0.75, 0.78125, 0.66875, 0.76875, 0.70625, 0.725, 0.73125, 0.79375, 0.73125, 0.76875], [0.738889, 0.7722220000000001, 0.8, 0.805556, 0.761111, 0.7888890000000001, 0.777778, 0.744444, 0.777778, 0.744444], [0.785, 0.705, 0.8, 0.79, 0.8, 0.805, 0.76, 0.775, 0.745, 0.83], [0.85, 0.8225, 0.8325, 0.825, 0.81, 0.8275, 0.8175, 0.79, 0.83, 0.7875], [0.821667, 0.816667, 0.83, 0.833333, 0.805, 0.816667, 0.801667, 0.835, 0.825, 0.826667], [0.835, 0.835, 0.8275, 0.84375, 0.8275, 0.8175, 0.83625, 0.84, 0.825, 0.8325]]

n = 10


test_scores_mean = np.mean(test_scores, axis=1)

# test_scores_mean = [float(Decimal("%.3f" % element)) for element in test_scores_mean]
test_scores_mean = list(np.around(np.array(test_scores_mean),3))

test_scores_std = np.std(test_scores, axis=1)
test_scores_std = list(np.around(np.array(test_scores_std),3))

test_scores_stdfailure = test_scores_std / np.sqrt(n)
test_scores_stdfailure = list(np.around(np.array(test_scores_stdfailure),3))
# test_scores_stdfailure = [float(Decimal("%.5f" % element)) for element in test_scores_stdfailure]
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

# df = pd.DataFrame({'mean':test_scores_mean})
df = pd.DataFrame({'mean':test_scores_mean, 'std':test_scores_std, 'stdfailure':test_scores_stdfailure, 'KIVneg':KIVneg, 'KIVpos':KIVpos})

print (df)

df.to_csv('table.csv')


