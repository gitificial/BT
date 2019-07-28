import numpy as np
import matplotlib.pyplot as plt

train_sizes = [100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

train_scores = [[0.612500, 0.725000, 0.587500], [0.705000, 0.780000, 0.680000], [0.792500, 0.825000, 0.810000], [0.806667, 0.865000, 0.816667], [0.827500, 0.813750, 0.813750], [0.860625, 0.853750, 0.868125], [0.854167, 0.860833, 0.857917], [0.861562, 0.887812, 0.864688], [0.875500, 0.877000, 0.863500], [0.883958, 0.886458, 0.875417], [0.878036, 0.885179, 0.881429], [0.879687, 0.876094, 0.890938], [0.876667, 0.885694, 0.886111], [0.881500, 0.880875, 0.889625]]

test_scores = [[0.650000, 0.600000, 0.500000], [0.720000, 0.760000, 0.680000], [0.880000, 0.760000, 0.800000], [0.786667, 0.846667, 0.726667], [0.890000, 0.810000, 0.820000], [0.887500, 0.882500, 0.867500], [0.893333, 0.900000, 0.918333], [0.900000, 0.901250, 0.906250], [0.910000, 0.916000, 0.928000], [0.908333, 0.914167, 0.910833], [0.906429, 0.915714, 0.918571], [0.920000, 0.915625, 0.913125], [0.920000, 0.913889, 0.916667], [0.924000, 0.911500, 0.921500]]

train_scores_mean = np.mean(train_scores, axis=1)
# print(train_scores_mean)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Test")

plt.xlabel("Datensätze")
plt.ylabel("Korrektklassifizierungsrate")

# plt.legend(loc="best")
plt.legend(loc="lower right")

plt.savefig('results.png')
plt.show()
