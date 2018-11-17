import numpy as np
import time


def dataLoader(datapath, labelpath):
    data = np.loadtxt(datapath)
    label = np.loadtxt(labelpath, int)
    _, indices, counts = np.unique(
        label, return_index=True, return_counts=True)

    # print some statistics to screen
    print("\nOriginal datset is: (before dropping Class 0)\n")
    for ind, cou in dict(zip(indices, counts)).items():
        print("Class {:d}: {:d} instances, {:1.3f}%".format(
            label[ind], cou, cou / data.shape[0] * 100))
    return data, label


def dropClass0(data, label):
    dataWithLabel = np.hstack((data, label[:, np.newaxis]))
    dataframe = dataWithLabel[dataWithLabel[:, -1] != 0]
    dataframe[:, -1] = dataframe[:, -1] - 1  # label is 0-based
    return dataframe


def downSample(dataframe, minSamples=None):
    # count the occurances of each class
    labels = dataframe[:, -1].astype(int)
    _, indices, counts = np.unique(
        labels, return_index=True, return_counts=True)
    numSamples = np.min(counts)
    if minSamples is not None:
        numSamples = minSamples
    uniqueLabels = labels[indices]
    # print("The minimal number of a class instances is: {}".format(numSamples))

    retDataFrame = randomSample(dataframe[labels == uniqueLabels[0]],
                                numSamples)

    for i in range(1, len(counts)):
        newDataframe = randomSample(dataframe[labels == uniqueLabels[i]],
                                    numSamples)
        retDataFrame = np.concatenate((retDataFrame, newDataframe), axis=0)

    # print some statistics to screen
    print("\nAfter dropping Class 0 and downsample the dataset:\n")
    retLabel = retDataFrame[:, -1].astype(int)
    _, indices, counts = np.unique(
        retLabel, return_index=True, return_counts=True)
    for ind, cou in dict(zip(indices, counts)).items():
        print("Class {:d}: {:d} instances, {:1.3f}%".format(
            retLabel[ind], cou, cou / retDataFrame.shape[0] * 100))

    return retDataFrame


def randomSample(dataframe, numSamples):
    indexMask = np.random.randint(dataframe.shape[0], size=numSamples)
    newDataframe = dataframe[indexMask]
    return newDataframe


def trainValSplit(dataframe, ratio=0.8):
    numClasses = np.unique(dataframe[:, -1]).shape[0]
    numSamples = int(dataframe.shape[0] / numClasses)
    trainNum = int(numSamples * ratio)
    trainIndexMask = np.random.randint(numSamples, size=trainNum)
    valIndexMask = np.array(range(numSamples), bool)
    valIndexMask[trainIndexMask] = False

    labels = dataframe[:, -1].astype(int)
    _, indices = np.unique(labels, return_index=True)
    uniqueLabels = labels[indices]

    tmp = dataframe[dataframe[:, -1] == uniqueLabels[0]]
    trainDataframe = tmp[trainIndexMask]
    valDataframe = tmp[valIndexMask]

    for i in range(1, numClasses):
        tmp = dataframe[dataframe[:, -1] == uniqueLabels[i]]
        trainDataframe = np.vstack((trainDataframe, tmp[trainIndexMask]))
        valDataframe = np.vstack((valDataframe, tmp[valIndexMask]))

    return (trainDataframe[:, :-1], trainDataframe[:, -1],
            valDataframe[:, :-1], valDataframe[:, -1])


def dataPreprocessor(datapath, labelpath, saveData=True):
    data, label = dataLoader(datapath, labelpath)
    # data = np.loadtxt(datapath)
    # label = np.loadtxt(labelpath, dtype=int)
    dataframe = dropClass0(data, label)
    dataframe = downSample(dataframe, 100)
    trainData, trainLabel, valData, valLabel = trainValSplit(dataframe)
    if saveData:
        import os
        # datadir = os.path.dirname(datapath)
        filename = os.path.splitext(datapath)[0]
        np.savetxt(
            filename + '_trainData.txt',
            trainData,
            fmt='%1.3f %1.3f %1.3f %d %d %d %d')
        np.savetxt(
            filename + '_valData.txt',
            valData,
            fmt='%1.3f %1.3f %1.3f %d %d %d %d')
        # labeldir = os.path.dirname(labelpath)
        filename = os.path.splitext(labelpath)[0]
        np.savetxt(filename + '_train.labels', trainLabel, fmt='%d')
        np.savetxt(filename + '_val.labels', valLabel, fmt='%d')
    return trainData, trainLabel, valData, valLabel


def main():
    print("Running ...")
    start = time.time()
    dataPreprocessor(
        r'./datasets/bildstein_station1_xyz_intensity_rgb.txt',
        r'./datasets/bildstein_station1_xyz_intensity_rgb.labels')
    # r'./datasets/for_script_testing.txt',
    # r'./datasets/for_script_testing.labels')
    end = time.time()
    print("Done in {:1.2f}s!".format(end - start))


if __name__ == '__main__':
    main()
