import numpy as np


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def sample_spherical_2(npoints, r):
    theta = np.random.uniform(0, 2 * np.pi, npoints)
    a = np.random.uniform(-1, 1, npoints)
    phi = np.arccos(a)
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)
    return x, y, z


# Make sphere
xi, yi, zi = sample_spherical_2(1000000, 1)
xi += 1
yi += 1
zi += 1

df_sphere = np.hstack((xi[:, np.newaxis], yi[:, np.newaxis],
                       zi[:, np.newaxis]))
ii = np.zeros_like(xi, dtype=int)[:, np.newaxis]
# random color
ri = np.random.randint(100, 110, (1000000, 1))
gi = np.random.randint(150, 160, (1000000, 1))
bi = np.random.randint(80, 90, (1000000, 1))

sphere = np.hstack((df_sphere, ii, ri, gi, bi))

# np.savetxt('./TestEnv/sphere.txt', sphere, fmt='%1.3f %1.3f %1.3f %d %d %d %d')

# Make plane
xx = np.linspace(5, 10, 1000)
yy = np.linspace(5, 10, 1000)
nx, ny = np.meshgrid(xx, yy)
xy = np.vstack((nx.ravel(), ny.ravel())).T
zz = np.zeros_like(nx.ravel())[:, np.newaxis] + np.random.normal(
    0, 0.05, (1000000, 1)) - 1

rri = np.random.randint(200, 210, (1000000, 1))
ggi = np.random.randint(110, 120, (1000000, 1))
bbi = np.random.randint(30, 40, (1000000, 1))

plane = np.hstack((xy, zz, ii, rri, ggi, bbi))
df = np.vstack((sphere, plane))
np.savetxt(
    './TestEnv/synthetic_testset2.txt',
    df,
    fmt='%1.3f %1.3f %1.3f %d %d %d %d')

# labels = np.vstack((np.zeros((1000000, 1)), np.ones((1000000, 1))))
# np.savetxt('./TestEnv/synthetic_testset.labels', labels, fmt='%d')
