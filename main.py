import numpy as np
import pandas as pd
from NN import NeuralNetwork


def ReadFile(FileName):
    line = np.array(pd.read_csv(FileName, header=None, delim_whitespace=True, nrows=2))
    data = pd.read_csv(FileName, skiprows=[0, 1], header=None, delim_whitespace=True)

    Ninput, Nhidden, NOutput, Nsapmles = int(line[0][0]), int(line[0][1]), int(line[0][2]), int(line[1][0])

    X_data = (data.iloc[:, 0:Ninput])
    X_data = (X_data - X_data.mean()) / X_data.std()

    X = np.array(X_data)
    Y = np.array((data.iloc[:, Ninput:]) / data.iloc[:, Ninput:].max())

    return Ninput, Nhidden, NOutput, Nsapmles, X, Y


Ninput, Nhidden, NOutput, Nsapmles, X, Y = ReadFile("input.txt")
alpha = 0.005
n_iterations = 500
# generate the wight randomly
HiddenW = np.random.randn(Ninput, Nhidden)
OutputW = np.random.randn(Nhidden, NOutput)

HiddenW, OutputW, error, y = NeuralNetwork(Nhidden, NOutput, Nsapmles, alpha, X, Y, n_iterations, HiddenW, OutputW)

HiddenW, OutputW, error, y = NeuralNetwork(Nhidden, NOutput, Nsapmles, alpha, X, Y, 1, HiddenW, OutputW)
print(error)
print(HiddenW)
print(OutputW)
file = open("output", "x")
file.write("Hidden Wights= " + str(HiddenW) + "\n")
file.write("Output Wights= " + str(OutputW) + "\n")
file.write("Error= " + str(error))
