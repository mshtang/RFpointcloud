import numpy as np
from sklearn.cluster import KMeans


def dataLoader(datapath, labelpath):
    print('Loading data ... ')
    data = np.loadtxt(datapath)
    label = np.loadtxt(labelpath, int)
    print('{:d} points and {:d} labels are read'.format(
        data.shape[0], label.shape[0]))
    return data, label


def makeClusters(dataframe, filename):
    coords = dataframe[:, :3]
    n_cluster = 2
    kmeans = KMeans(n_cluster, 'random', 1, 10, n_jobs=-1).fit(coords)
    clusters = kmeans.labels_
    for i in range(n_cluster):
        part = dataframe[clusters == i]
        print("size of part {:d}: {:d}".format(i, part.shape[0]))
        np.savetxt(
            './datasets/{}_part_{:d}.txt'.format(filename, i),
            part[:, :-1],
            fmt='%1.3f %1.3f %1.3f %d %d %d %d')
        np.savetxt(
            './datasets/{}_part_{:d}.labels'.format(filename, i),
            part[:, -1],
            fmt='%d')


if __name__ == "__main__":
    data, labels = dataLoader(
        r'./datasets/bildstein_station1_xyz_intensity_rgb_dropped.txt',
        r'./datasets/bildstein_station1_xyz_intensity_rgb_dropped.labels')
    # r'./TestEnv/downsampled.txt',
    # r'./TestEnv/downsampled.labels')
    dataframe = np.hstack((data, labels[:, np.newaxis]))
    for i in range(8):
        df = dataframe[dataframe[:, -1] == i]
        makeClusters(df, 'class_' + str(i))
