import numpy as np
import time
import os

np.random.seed(123)


def dataLoader(datapath, labelpath):
    data = np.loadtxt(datapath)
    label = np.loadtxt(labelpath, int)
    return data, label


def reportStat(dataframe, operation):
    """
    print some statistics to screen
    """
    label = dataframe[:, -1].astype(int)
    _, indices, counts = np.unique(
        label, return_index=True, return_counts=True)

    print("\n {} contains points of {:d}:\n".format(operation,
                                                    dataframe.shape[0]))
    for ind, cou in dict(zip(indices, counts)).items():
        print("Class {:d}: {:d} instances, {:1.3f}%".format(
            label[ind], cou, cou / dataframe.shape[0] * 100))


def dropClass0(dataframe):
    dataframe = dataframe[dataframe[:, -1] != 0]
    dataframe[:, -1] = dataframe[:, -1] - 1  # label is 0-based
    reportStat(dataframe, "After dropping Class 0")
    return dataframe


def randomSample(dataframe, numSamples, return_mask=False):
    # if numSamples is less than one ,then this is a ratio
    if numSamples < 1:
        numSamples = int(dataframe.shape[0] * numSamples)
    # if numSamples is very small , the product can be less than 1
    # then set the minimal number of samples to 1
    if numSamples < 1:
        numSamples = 1
    indexMask = np.random.choice(dataframe.shape[0], numSamples, replace=False)
    newDataframe = dataframe[indexMask]
    if not return_mask:
        return newDataframe
    return newDataframe, indexMask


def downSample(dataframe, minSamples=None):
    """
    To downsample the data, if minSamples not given, the minSamples is the number of samples
    in the least class, if minSamples larger than 1, use this value to sample each class, 
    otherwise treat this number as a ratio.
    """
    # count the occurances of each class
    labels = dataframe[:, -1].astype(int)
    _, indices, counts = np.unique(
        labels, return_index=True, return_counts=True)
    uniqueLabels = labels[indices]
    numSamples = int(np.min(counts))
    if minSamples is None:
        print("The minimal number of a class instances is: {}".format(
            numSamples))
    else:
        numSamples = minSamples

    retDataFrame = randomSample(dataframe[labels == uniqueLabels[0]],
                                numSamples)

    for i in range(1, len(counts)):
        newDataframe = randomSample(dataframe[labels == uniqueLabels[i]],
                                    numSamples)
        retDataFrame = np.vstack((retDataFrame, newDataframe))

    reportStat(retDataFrame, "After dropping Class 0 and downsample")

    return retDataFrame


def trainValSplit(dataframe, ratio=0.8):
    labels = dataframe[:, -1].astype(int)
    _, indices = np.unique(labels, return_index=True)
    uniqueLabels = labels[indices]
    numClasses = uniqueLabels.shape[0]

    tmp = dataframe[labels == uniqueLabels[0]]
    trainDataframe, trainIndexMask = randomSample(tmp, ratio, True)
    valIndexMask = np.full(tmp.shape[0], True)
    valIndexMask[trainIndexMask] = False
    valDataframe = tmp[valIndexMask]

    for i in range(1, numClasses):
        tmp = dataframe[labels == uniqueLabels[i]]
        tmpTrainDataframe, trainIndexMask = randomSample(tmp, ratio, True)
        valIndexMask = np.full(tmp.shape[0], True)
        valIndexMask[trainIndexMask] = False
        trainDataframe = np.vstack((trainDataframe, tmpTrainDataframe))
        valDataframe = np.vstack((valDataframe, tmp[valIndexMask]))

    reportStat(trainDataframe, "The trainset")
    reportStat(valDataframe, "The valset")

    return (trainDataframe[:, :-1], trainDataframe[:, -1],
            valDataframe[:, :-1], valDataframe[:, -1])


def dataPreprocessor(datapath, labelpath):
    filename = os.path.splitext(datapath)[0]

    if os.path.isfile(filename + '_dropped.txt'):
        data = np.loadtxt(filename + '_dropped.txt')
        label = np.loadtxt(filename + '_dropped.labels')
        dataframe = np.hstack((data, label[:, np.newaxis]))
        reportStat(dataframe, "After dropping Class 0")
    else:
        data, label = dataLoader(datapath, labelpath)
        dataframe = np.hstack((data, label[:, np.newaxis]))
        reportStat(dataframe, "Original dataset")

        dataframe = dropClass0(dataframe)
        np.savetxt(
            filename + '_dropped.txt',
            dataframe[:, :-1],
            fmt='%1.3f %1.3f %1.3f %d %d %d %d')
        np.savetxt(filename + '_dropped.labels', dataframe[:, -1], fmt='%d')
    ##########################
    # CHANGE THE RATIO HERE  #
    ##########################
    dataframe = downSample(dataframe, 1000)
    np.savetxt(
        filename + '_downsample_100.txt',
        dataframe[:, :-1],
        fmt='%1.3f %1.3f %1.3f %d %d %d %d')
    np.savetxt(filename + '_downsample_100.labels', dataframe[:, -1], fmt='%d')

    # trainData, trainLabel, valData, valLabel = trainValSplit(dataframe)
    # np.savetxt(
    #     filename + '_trainData.txt',
    #     trainData,
    #     fmt='%1.3f %1.3f %1.3f %d %d %d %d')
    # np.savetxt(
    #     filename + '_valData.txt',
    #     valData,
    #     fmt='%1.3f %1.3f %1.3f %d %d %d %d')
    # filename = os.path.splitext(labelpath)[0]
    # np.savetxt(filename + '_train.labels', trainLabel, fmt='%d')
    # np.savetxt(filename + '_val.labels', valLabel, fmt='%d')
    # return trainData, trainLabel, valData, valLabel


def main():
    print("Running ...")
    start = time.time()
    dataPreprocessor(
        r'./datasets/bildstein_station1_xyz_intensity_rgb.txt',
        r'./datasets/bildstein_station1_xyz_intensity_rgb.labels')
    # r'./datasets/for_script_testing.txt',
    # r'./datasets/for_script_testing.labels')
    end = time.time()
    print("\nDone in {:1.2f}s!".format(end - start))


if __name__ == '__main__':
    main()