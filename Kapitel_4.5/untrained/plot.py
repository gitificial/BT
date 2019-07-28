import numpy as np
import matplotlib.pyplot as plt

train_sizes = [100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]


train_scores = [[0.875, 0.6375, 0.7625], [0.84, 0.535, 0.82], [0.82, 0.805, 0.8375], [0.821666665871938, 0.826666665871938, 0.8183333325386047], [0.82125, 0.805, 0.83375], [0.771875, 0.7725, 0.780625], [0.7675, 0.8008333333333333, 0.78375], [0.7746875, 0.7796875, 0.77625], [0.78375, 0.76775, 0.78625], [0.7841666666666667, 0.769375, 0.7820833333333334], [0.7730357142857143, 0.7626785714285714, 0.7725], [0.7765625, 0.77796875, 0.78265625], [0.7702777777777777, 0.7738888888888888, 0.7726388888888889], [0.76425, 0.771875, 0.75975]]

test_scores = [[0.800000011920929, 0.6499999761581421, 0.6499999761581421], [0.7200000286102295, 0.6200000047683716, 0.7599999904632568], [0.7600000095367432, 0.7799999928474426, 0.8000000047683716], [0.8000000015894572, 0.7933333365122477, 0.7800000015894571], [0.83, 0.81, 0.835], [0.845, 0.8475, 0.825], [0.8766666658719381, 0.8433333325386048, 0.8183333333333334], [0.83875, 0.84, 0.83125], [0.846, 0.8389999990463257, 0.8359999995231628], [0.8441666666666666, 0.8400000007947286, 0.8491666658719381], [0.8635714295932225, 0.8192857139451163, 0.8285714278902326], [0.840625, 0.843125, 0.86], [0.8722222222222222, 0.8722222222222222, 0.8605555555555555], [0.862, 0.856, 0.863]]

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
