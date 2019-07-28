import numpy as np
import matplotlib.pyplot as plt

test_scores_rand = [[0.699999988079071, 0.800000011920929, 0.699999988079071], [0.8600000143051147, 0.800000011920929, 0.8199999928474426], [0.9099999976158142, 0.8399999976158142, 0.8399999928474426], [0.8600000015894572, 0.8933333309491476, 0.8666666642824808], [0.9, 0.905, 0.925], [0.9175, 0.935, 0.9275], [0.918333334128062, 0.9100000007947286, 0.9083333333333333], [0.90375, 0.9375, 0.93]]

test_scores_static = [[0.699999988079071, 0.800000011920929, 0.6499999761581421], [0.7400000095367432, 0.6800000071525574, 0.7599999904632568], [0.7099999904632568, 0.8099999928474426, 0.8299999976158142], [0.7266666658719381, 0.8666666634877522, 0.7866666658719381], [0.84, 0.815, 0.825], [0.8675, 0.8825, 0.8725], [0.9166666674613952, 0.8850000007947286, 0.8883333325386047], [0.88375, 0.91875, 0.90375]]


test_scores_nstatic = [[0.699999988079071, 0.75, 0.6499999761581421], [0.8799999952316284, 0.8600000143051147, 0.8399999737739563], [0.8699999976158143, 0.8900000023841858, 0.8999999976158142], [0.8800000015894572, 0.8933333309491476, 0.860000003973643], [0.885, 0.875, 0.895], [0.91, 0.9325, 0.9275], [0.9350000007947286, 0.9183333325386047, 0.913333334128062], [0.9025, 0.94125, 0.92375]]


#---------------------------------------------------------------------------------

train_sizes = [100, 250, 500, 750, 1000, 2000, 3000, 4000]

# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)

# CNN-rand
test_scores_mean_rand = np.mean(test_scores_rand, axis=1)
test_scores_std_rand = np.std(test_scores_rand, axis=1)

# CNN-static
test_scores_mean_static = np.mean(test_scores_static, axis=1)
test_scores_std_static = np.std(test_scores_static, axis=1)

# CNN-non-static
test_scores_mean_nstatic = np.mean(test_scores_nstatic, axis=1)
test_scores_std_nstatic = np.std(test_scores_nstatic, axis=1)
plt.grid()

# plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
# plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")

# CNN-rand
plt.fill_between(train_sizes, test_scores_mean_rand - test_scores_std_rand, test_scores_mean_rand + test_scores_std_rand, alpha=0.1, color="g")
plt.plot(train_sizes, test_scores_mean_rand, 'o-', color="g", label="CNN-rand Test")

# CNN-static
plt.fill_between(train_sizes, test_scores_mean_static - test_scores_std_static, test_scores_mean_static + test_scores_std_static, alpha=0.1, color="b")
plt.plot(train_sizes, test_scores_mean_static, 'o-', color="b", label="CNN-static Test")

# CNN-non-static
plt.fill_between(train_sizes, test_scores_mean_nstatic - test_scores_std_nstatic, test_scores_mean_nstatic + test_scores_std_nstatic, alpha=0.1, color="r")
plt.plot(train_sizes, test_scores_mean_nstatic, 'o-', color="r", label="CNN-non-static Test")

plt.xlabel("Datensätze")
# plt.ylabel("Genauigkeit")
plt.ylabel("Korrektklassifizierungsrate")
# plt.title("Test Results", fontdict=None, loc='center')

plt.legend(loc="best")
plt.savefig('results.png')
plt.show()


#---------------------------------------------------------------------------------
