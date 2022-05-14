import pandas as pd
import numpy as np
import PIL
import pywt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import matplotlib.pyplot as plt

# np.random.seed(5)

df = pd.read_excel("ImageLabels.xlsx")
# df = df[df["Class"]!=3]
print(df.shape)

df["image"] = df.ID.apply(lambda idx: np.array(PIL.Image.open("Images/D"+str(idx)+"_icon.jpg").getdata())[:, 1])
df["ch"] = df["image"].apply(lambda img:
  pywt.dwt2(img.reshape(75, 75), 'haar')[1][2].flatten())
# cA, (cH, cV, cD) = pywt.dwt2(df["image"][0].reshape(75, 75), 'haar')
# im = PIL.Image.fromarray(np.uint8(255 * cH / np.max(cH)))
# im.show()

runs = 100
m_array = [1, 2, 4, 8, 10, 15, 20, 25, 35, 40, 50, 70, 80, 100, 150, 200]
r1, r2, r3, r4 = np.zeros((runs, len(m_array))), np.zeros((runs, len(m_array))), np.zeros((runs, len(m_array))), np.zeros((runs, len(m_array)))
for r in range(runs):
  for j, m in enumerate(m_array):
    X = np.stack(df.ch.to_numpy())
    y = df.Class.to_numpy()
    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    y = y[perm]
    A = np.random.binomial(1, 0.5, size=(m, X[0].size))
    A = (2*A - 1)/np.sqrt(m)
    num_test = int(X.shape[0] * 0.5)

    X_train = X[:num_test]
    X_train_comp = (A @ X_train.T).T
    y_train = y[:num_test]

    X_test = X[num_test:]
    X_test_comp = (A @ X_test.T).T
    y_test = y[num_test:]

    C=1.4
    clf = make_pipeline(SVC(C=C))
    clf.fit(X_train, y_train)

    r1[r, j] = (clf.score(X_train, y_train))
    r2[r, j] = (clf.score(X_test, y_test))

    # print(X_train_comp.shape, X_test_comp.shape)
    clf_comp = make_pipeline(SVC(C=C))
    clf_comp.fit(X_train_comp, y_train)

    r3[r, j] = (clf_comp.score(X_train_comp, y_train))
    r4[r, j] = (clf_comp.score(X_test_comp, y_test))

print(r1, r2)
plt.plot(m_array, r1.mean(axis=0), label="Uncompressive")
plt.plot(m_array, r3.mean(axis=0), label="Compressive")
plt.legend()
plt.show()
plt.plot(m_array, r2.mean(axis=0), label="Uncompressive")
plt.plot(m_array, r4.mean(axis=0), label="Compressive")
plt.legend()
plt.show()