import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.externals import joblib

"""
实验：
n_components分别取256、512、1024，考察信息保留情况
"""
# X = pd.read_table('../out/b.txt', header=None).values
# pca = PCA(n_components=1024, svd_solver='randomized')
# pca.fit(X)
# print(sum(pca.explained_variance_ratio_))

"""
实验结果：
n_components=256，信息保留0.8084744650204365
n_components=512，信息保留0.8995493804453463
n_components=1024，信息保留0.9631881579233071
"""

"""
增量训练：
"""
n_components = 256
batch_size = 2048
file_input = '../out/a.txt'
file_out = '../out/model_photo_visual_pca'
with open(file_input, 'r') as f_in:
    X_batch = []
    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size, whiten=False)
    for (num, line) in enumerate(f_in):
        line = line.replace("\n", "")
        X_batch.append([float(x) for x in line.split('\t')])
        if (num + 1) % batch_size == 0:
            print(num + 1, flush=True)
            pca.partial_fit(np.array(X_batch))
            X_batch.clear()
    pca.partial_fit(np.array(X_batch))
    X_batch.clear()
    joblib.dump(pca, file_out)
