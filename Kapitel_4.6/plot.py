import numpy as np
import matplotlib.pyplot as plt

train_sizes = [100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

random_words = [[0.699999988079071, 0.6499999761581421, 0.699999988079071], [0.6200000047683716, 0.6000000238418579, 0.6800000071525574], [0.6300000023841857, 0.6500000071525573, 0.8099999928474426], [0.686666665871938, 0.7666666634877523, 0.6733333333333333], [0.74, 0.78, 0.735], [0.7475, 0.8, 0.775], [0.8483333325386048, 0.8433333333333334, 0.8100000007947286], [0.87625, 0.85, 0.84625], [0.861, 0.87, 0.8689999990463256], [0.8933333325386047, 0.884166665871938, 0.8883333333333333], [0.8642857149669102, 0.8792857149669102, 0.8764285714285714], [0.8725, 0.878125, 0.88625], [0.8688888888888889, 0.9088888888888889, 0.8888888888888888], [0.8985, 0.906, 0.8845]]

test_scores_trained = [[0.650000, 0.600000, 0.500000], [0.720000, 0.760000, 0.680000], [0.880000, 0.760000, 0.800000], [0.786667, 0.846667, 0.726667], [0.890000, 0.810000, 0.820000], [0.887500, 0.882500, 0.867500], [0.893333, 0.900000, 0.918333], [0.900000, 0.901250, 0.906250], [0.910000, 0.916000, 0.928000], [0.908333, 0.914167, 0.910833], [0.906429, 0.915714, 0.918571], [0.920000, 0.915625, 0.913125], [0.920000, 0.913889, 0.916667], [0.924000, 0.911500, 0.921500]]

test_scores_untrained = [[0.800000011920929, 0.6499999761581421, 0.6499999761581421], [0.7200000286102295, 0.6200000047683716, 0.7599999904632568], [0.7600000095367432, 0.7799999928474426, 0.8000000047683716], [0.8000000015894572, 0.7933333365122477, 0.7800000015894571], [0.83, 0.81, 0.835], [0.845, 0.8475, 0.825], [0.8766666658719381, 0.8433333325386048, 0.8183333333333334], [0.83875, 0.84, 0.83125], [0.846, 0.8389999990463257, 0.8359999995231628], [0.8441666666666666, 0.8400000007947286, 0.8491666658719381], [0.8635714295932225, 0.8192857139451163, 0.8285714278902326], [0.840625, 0.843125, 0.86], [0.8722222222222222, 0.8722222222222222, 0.8605555555555555], [0.862, 0.856, 0.863]]

random_words_mean = np.mean(random_words, axis=1)
random_words_std = np.std(random_words, axis=1)

test_scores_trained_mean = np.mean(test_scores_trained, axis=1)
test_scores_trained_std = np.std(test_scores_trained, axis=1)

test_scores_untrained_mean = np.mean(test_scores_untrained, axis=1)
test_scores_untrained_std = np.std(test_scores_untrained, axis=1)
plt.grid()

plt.fill_between(train_sizes, random_words_mean - random_words_std, random_words_mean + random_words_std, alpha=0.1, color="g")

plt.fill_between(train_sizes, test_scores_trained_mean - test_scores_trained_std, test_scores_trained_mean + test_scores_trained_std, alpha=0.1, color="darkgrey")

plt.fill_between(train_sizes, test_scores_untrained_mean - test_scores_untrained_std, test_scores_untrained_mean + test_scores_untrained_std, alpha=0.1, color="darkgrey")

plt.plot(train_sizes, random_words_mean, 'o-', color="g", label="randomisierte Substantive")
plt.plot(train_sizes, test_scores_trained_mean, 's-', color="darkgrey", label="trainierte Einbettung")
plt.plot(train_sizes, test_scores_untrained_mean, 'o-', color="darkgrey", label="untrainierte Einbettung")

plt.xlabel("Datensätze")
plt.ylabel("Korrektklassifizierungsrate")

# plt.legend(loc="best")
plt.legend(loc="lower right")

plt.savefig('results.png')
plt.show()
