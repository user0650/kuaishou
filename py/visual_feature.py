"""
视觉特征转为8聚类、16聚类、32聚类、64聚类、128聚类 ...
"""

import pandas as pd
import numpy as np

base_feature_csv = '../out/base_feature.csv'

train_visual_dir = '/gdata/Kwai2018Final/train/final_visual_train/'
test_visual_dir = '/gdata/Kwai2018Final/test/final_visual_test/'

photo_visual_cluster_dir = '../out/'
feature_dir = '../out/features/'
visual_feature_csv = '../out/features/f_visual.csv'


def export_visual_cluster_feature(n_clusters=8):
    file = photo_visual_cluster_dir + 'photo_visual_cluster_' + str(n_clusters) + '.csv'
    df_cluster = pd.read_csv(file)
    df_base = pd.read_csv(base_feature_csv)[['photo_id']]
    df = pd.merge(df_base, df_cluster, how='left', on='photo_id')
    df.drop(columns=['photo_id'], inplace=True)
    f_name = 'f_visual_cluster_' + str(n_clusters)
    df.columns = [f_name]
    df.to_csv(feature_dir + f_name + '.csv', index=False, header=True)


def get_visual_data(photo_id, _flag_):
    """
    读取一个图片的视觉特征
    """
    if _flag_ >= 0:
        return ",".join(str(x) for x in np.load(train_visual_dir + str(photo_id))[0])
    else:
        return ",".join(str(x) for x in np.load(test_visual_dir + str(photo_id))[0])


def export_visual_feature():
    df_base = pd.read_csv(base_feature_csv)
    with open(visual_feature_csv + '', 'w') as file_out:
        # 写表头：
        file_out.write(','.join(('f_visual_' + str(i)) for i in range(2048)) + '\n')
        # 写数据：
        df_base.apply(lambda row: file_out.write(get_visual_data(row['photo_id'], row['_flag_']) + '\n'), axis=1)


if __name__ == '__main__':
    export_visual_feature()
