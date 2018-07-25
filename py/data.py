"""
data processing
"""

import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib

train_interaction_txt = '/gdata/Kwai2018Final/train/train_interaction.txt'
test_interaction_txt = '/gdata/Kwai2018Final/test/test_interaction.txt'
train_visual_dir = '/gdata/Kwai2018Final/train/final_visual_train/'
test_visual_dir = '/gdata/Kwai2018Final/test/final_visual_test/'

photo_visual_txt = '../out/photo_visual.txt'
model_cluster_dir = '../model/'
photo_visual_dir = '../out/'


def export_photo_visual_data():
    """
    export photo visual data:
    photo_id_1 feature1 feature2 ... feature2048
    ...
    photo_id_n feature1 feature2 ... feature2048
    :return:
    """
    df_train = pd.read_table(train_interaction_txt, header=None)
    photo_train = df_train[1].unique()
    df_test = pd.read_table(test_interaction_txt, header=None)
    photo_test = df_test[1].unique()
    with open(photo_visual_txt, 'w') as file_out:
        for photo_id in photo_train:
            features = "\t".join(str(x) for x in np.load(train_visual_dir + str(photo_id))[0])
            file_out.write(str(photo_id) + '\t' + features + '\n')
        for photo_id in photo_test:
            features = "\t".join(str(x) for x in np.load(test_visual_dir + str(photo_id))[0])
            file_out.write(str(photo_id) + '\t' + features + '\n')


def train_photo_visual_data(n_clusters=8):
    """
    图片视觉特征聚类
    :return:
    """
    batch_size = 8192
    model_file = model_cluster_dir + 'model_photo_visual_cluster_' + str(n_clusters)

    with open(photo_visual_txt, 'r') as f_in:
        X_batch = []
        cls = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size)
        for (num, line) in enumerate(f_in):
            line = line.replace("\n", "")
            row = [float(x) for x in line.split('\t')[1:]]
            X_batch.append(row)
            if (num + 1) % batch_size == 0:
                print('cluster: ', num + 1, flush=True)
                cls.partial_fit(np.array(X_batch))
                X_batch.clear()
        if len(X_batch) >= n_clusters:
            cls.partial_fit(np.array(X_batch))
            X_batch.clear()
        joblib.dump(cls, model_file)


def cluster_photo_visual_data(n_clusters=8):
    """
    图片视觉特征聚类
    :param n_clusters:
    :return:
    """
    model_file = model_cluster_dir + 'model_photo_visual_cluster_' + str(n_clusters)
    out_file = photo_visual_dir + 'photo_visual_cluster_' + str(n_clusters) + '.csv'

    batch_size = 8192
    X_batch = []
    id_batch = []
    with open(photo_visual_txt, 'r') as f_in, open(out_file, 'w') as f_out:
        f_out.write('photo_id,cluster_' + str(n_clusters) + '\n')  # 表头
        cls = joblib.load(model_file)
        for (num, line) in enumerate(f_in):
            line = line.replace("\n", "")
            row = [float(x) for x in line.split('\t')]
            photo_id = int(row[0])
            photo_data = row[1:]
            id_batch.append(photo_id)
            X_batch.append(photo_data)
            if (num + 1) % batch_size == 0:
                print('predict: ', num + 1, flush=True)
                y_batch = cls.predict(X_batch)
                for i in range(len(id_batch)):
                    f_out.write(str(id_batch[i]) + ',' + str(y_batch[i]) + '\n')
                id_batch.clear()
                X_batch.clear()
        y_batch = cls.predict(X_batch)
        for i in range(len(id_batch)):
            f_out.write(str(id_batch[i]) + ',' + str(y_batch[i]) + '\n')
        id_batch.clear()
        X_batch.clear()


if __name__ == '__main__':
    # export_photo_visual_data()
    # train_photo_visual_data(8)
    cluster_photo_visual_data(8)
