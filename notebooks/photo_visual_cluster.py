import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib

"""
增量训练：
"""
with open('../out/a.txt', 'r') as f_in:
    X_batch = []
    batch_size = 2048
    cls = MiniBatchKMeans(n_clusters=128, batch_size=batch_size)
    for (num, line) in enumerate(f_in):
        line = line.replace("\n", "")
        row = [float(x) for x in line.split('\t')]
        X_batch.append(row)
        if (num + 1) % batch_size == 0:
            print(np.array(X_batch).shape)
            cls.partial_fit(np.array(X_batch))
            X_batch.clear()
    print(np.array(X_batch).shape)
    cls.partial_fit(np.array(X_batch))
    X_batch.clear()
    joblib.dump(cls, '../out/model_photo_visual_cluster.txt')
