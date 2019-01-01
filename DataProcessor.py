import os
import numpy as np
from sklearn.cluster import KMeans


def dataLoader(datapath, labelpath):
    print('Loading data ... ')
    data = np.loadtxt(datapath)
    label = np.loadtxt(labelpath, int)
    print('{:d} points and {:d} labels are read'.format(
        data.shape[0], label.shape[0]))
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

    return dataframe


def makeClusters(dataframe):
    coords = dataframe[:, :3]
    n_cluster = 10
    kmeans = KMeans(n_cluster, 'random', 1, 10, n_jobs=-1).fit(coords)
    clusters = kmeans.labels_
    parts = [dataframe[clusters == i] for i in range(n_cluster)]
    return parts


if __name__ == "__main__":
    dataframe = dataPreprocessor(
        r'./datasets/other datasets/bildstein_station3_xyz_intensity_rgb.txt',
        r'./datasets/other datasets/bildstein_station3_xyz_intensity_rgb.labels'
    )

    all_parts = []
    for i in range(8):
        df = dataframe[dataframe[:, -1] == i]
        parts = makeClusters(df)
        all_parts.append(parts)

    for c in range(len(all_parts[i])):
        for r in range(len(all_parts)):
            newdata = all_parts[r][c]
            with open(
                    './datasets/testsets/{}_part_{:d}.txt'.format(
                        'bildstein3', c), 'a') as f1:
                np.savetxt(
                    f1, newdata[:, :-1], fmt='%1.3f %1.3f %1.3f %d %d %d %d')
            with open(
                    './datasets/testsets/{}_part_{:d}.labels'.format(
                        'bildstein3', c), 'a') as f2:
                np.savetxt(f2, newdata[:, -1], fmt='%d')
