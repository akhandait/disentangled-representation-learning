import pickle
import matplotlib.pyplot as plt
import numpy as np

dira = 'output_unrolled_mnist_6/'
dirab = 'output_unrolled_mnist_7/'
dirad = 'output_unrolled_mnist_8/'
dirac = 'output_mnist_2best/'
# out = 'output_mnist_2best/plots/'

# for i in range(6):
pickle_in = open(dira + '/lists.pickle', 'rb')
lists = pickle.load(pickle_in)
lossesD, lossesG, lossesQDis, lossesQCon, opt = lists

pickle_in = open(dirac + '/lists.pickle', 'rb')
listsc = pickle.load(pickle_in)
lossesDc, lossesGc, lossesQDisc, lossesQConc, optc = listsc
pickle_in = open(dirad + '/lists.pickle', 'rb')
listsd = pickle.load(pickle_in)
lossesDd, lossesGd, lossesQDisd, lossesQCond, optd = listsd
pickle_in = open(dirab + '/lists.pickle', 'rb')
listsb = pickle.load(pickle_in)
lossesDb, lossesGb, lossesQDisb, lossesQConb, optb = listsb
# print(opt)
# plt.axis([0, 10000, 0, 1000])
# print(opt)
# print(lossesQDis[-1])
# plt.plot()
print(lossesQDis[-1])
print(lossesQDisb[-1])
print(lossesQDisd[-1])
print(lossesQDisc[-1])

# lossesQCon = [i + 0.18 for i in lossesQCon]
# lossesQConb = [i + 0.18 for i in lossesQConb]
# lossesQCond = [i + 0.18 for i in lossesQCond]
# lossesQConc = [i + 0.18 for i in lossesQConc]
# t = np.linspace(0., 15., 82)

# plt.grid(True)
# plt.plot(t, lossesQCon[:16400:200], label='10 unrolling steps')
# plt.plot(t, lossesQConb[:16400:200], label='7 unrolling steps')
# plt.plot(t, lossesQCond[:16400:200], label='5 unrolling steps')
# plt.plot(t, lossesQConc[:16400:200], label='no unrolling')
# # plt.suptitle('Q loss for ')
# plt.legend(loc='upper right')
# plt.xlabel('Epochs')
# plt.ylabel('Q continuos latent variable loss(normalized)')
# plt.savefig(dira + 'lossesQCon.png')
# plt.close()


# pickle_in = open('output_resBlock/lists5.pickle', 'rb')
# lists = pickle.load(pickle_in)

# klLossList, reconLossList, samplesPriorList, opt = lists
# plt.plot(klLossList)
# plt.plot(reconLossList)
# plt.show()
# # print(len(klLossList))
