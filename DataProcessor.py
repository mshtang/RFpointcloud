import numpy as np
from sklearn.cluster import KMeans


def dataLoader(datapath, labelpath):
    data = np.loadtxt(datapath)
    label = np.loadtxt(labelpath, int)
    return data, label


def makeClusters(dataframe):
    coords = dataframe[:, :3]
    n_cluster = 5
    kmeans = KMeans(n_cluster, 'random', n_jobs=-1).fit(coords)
    clusters = kmeans.labels_
    for i in range(n_cluster):
        part = dataframe[clusters == i]
        print("size of part {:d}: {:d}".format(i, part.shape[0]))
        np.savetxt(
            './toy_dataset/part_{:d}.txt'.format(i),
            part[:, :-1],
            fmt='%1.3f %1.3f %1.3f %d %d %d %d')
        np.savetxt(
            './toy_dataset/part_{:d}.labels'.format(i), part[:, -1], fmt='%d')


if __name__ == "__main__":
    data, labels = dataLoader(
        r'./datasets/bildstein_station1_xyz_intensity_rgb_trainData.txt',
        r'./datasets/bildstein_station1_xyz_intensity_rgb_train.labels')
    dataframe = np.hstack((data, labels[:, np.newaxis]))
    makeClusters(dataframe)
