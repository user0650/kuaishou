# -*- coding: UTF-8 -*-

import numpy as np
from sklearn.externals import joblib

"""
用 model_photo_visual_pca 把 train_photo_visual.txt 降维
输出 train_photo_visual_pca.txt
"""

file_pca = '../out/model_photo_visual_pca'
file_input = '../out/a.txt'
file_out = '../out/a_pca.txt'
batch_size = 2048
with open(file_input, 'r') as f_in, open(file_out, 'w') as f_out:
    X_batch = []
    pca = joblib.load(file_pca)
    print(sum(pca.explained_variance_ratio_))
    for (num, line) in enumerate(f_in):
        line = line.replace("\n", "")
        X_batch.append([float(x) for x in line.split('\t')])
        if (num + 1) % batch_size == 0:
            print(num + 1)
            X_reduced = pca.transform(np.array(X_batch))
            # print(X_reduced.shape)
            f_out.write("\n".join("\t".join(str(x) for x in row) for row in X_reduced))
            f_out.write("\n")
            X_batch.clear()
    X_reduced = pca.transform(np.array(X_batch))
    # print(X_reduced.shape)
    f_out.write("\n".join("\t".join(str(x) for x in row) for row in X_reduced))
    f_out.write("\n")
    X_batch.clear()
