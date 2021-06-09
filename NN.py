import numpy as np


def Sigmoid(z):  # sigmoid function
    return 1 / (1 + (np.exp(-z)))


def OutputCalc(Wights, NeuronNum, X):  # calculate the output value for neuron NeuronNum
    sum = 0
    for i in range(np.size(X)):
        sum += Wights[i][NeuronNum] * X[i]
    return Sigmoid(sum)


def MSE(Target, Output):  # calculate Min Square Error
    sum = 0
    for i in range(np.size(Output)):
        sum += pow(Target[i] - Output[i], 2) * 1.0 / 2
    return sum


def FeedFrowerd(NOut, Nhidden, HiddenW, OutputW, X):
    NewY = []
    out = []
    # calculate the output value for each neroun at the hidden layer
    for neroun in range(Nhidden):
        out.append(OutputCalc(HiddenW, neroun, X))

    # calculate the output value (NewY) for each neroun at the output layer
    for neroun in range(NOut):
        NewY.append(OutputCalc(OutputW, neroun, out))

    return NewY, out


def UpdateWights(W, error, out, alpha):
    # update the wights by
    # w(new)=w(old)+(error(in)*alpha*out(out)
    for i in range(np.size(out)):
        for j in range(np.size(error)):
            W[i][j] = W[i][j] + (alpha * error[j] * out[i])


def BackPropagation(Nhiddens, Wight2, Y, NewY, out):
    ErrorHidden = []
    ErrorOutput = []
    for i in range(np.size(Y)):  # calc error for each neuron at output layer by
        # (OutY(1-OutY)*(TargetY-OutY)
        ErrorOutput.append(NewY[i] * (1 - NewY[i]) * (Y[i] - NewY[i]))

    for i in range(Nhiddens):  # calc error for each neuron at hidden layer
        sum = 0
        for j in range(np.size(ErrorOutput)):
            sum += ErrorOutput[j] * Wight2[i][j]
        ErrorHidden.append(sum * (1 - out[i]) * out[i])
    return ErrorOutput, ErrorHidden


MSEV = []


def main( Nhiddens, Nout, Nsamples, alpha, X, Y, Iters,HiddenW,OutputW):
    for iter in range(Iters):
        s = 0
        for i in range(np.size(Nsamples)):
            #feedFrowerd Step
            NewY, Out = FeedFrowerd(Nout, Nhiddens, HiddenW, OutputW, X[i])
            #BackPropagation Step
            ErrorOutput, ErrorHidden = BackPropagation(Nhiddens, HiddenW, Y[i], NewY, Out)
            #UpdateWights Step
            UpdateWights(OutputW, ErrorOutput, Out, alpha)
            UpdateWights(HiddenW, ErrorHidden, X[i], alpha)
            s += MSE(Y[i], NewY)
        MSEV.append(s / Nsamples)
    #print(MSEV)
    return HiddenW, OutputW,MSEV.pop()

