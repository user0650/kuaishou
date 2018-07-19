"""
视觉特征转为8聚类、16聚类、32聚类、64聚类、128聚类 ...
"""

import pandas as pd


photo_visual_cluster_dir = '../out/'
base_feature_csv = '../out/base_feature.csv'
feature_dir = '../out/features/'


def export_visual_feature(n_clusters=8):
    file = photo_visual_cluster_dir + 'photo_visual_cluster_' + str(n_clusters) + '.csv'
    df_cluster = pd.read_csv(file)
    df_base = pd.read_csv(base_feature_csv)[['photo_id']]
    df = pd.merge(df_base, df_cluster, how='left', on='photo_id')
    df.drop(columns=['photo_id'], inplace=True)
    f_name = 'f_visual_cluster_' + str(n_clusters)
    df.columns = [f_name]
    df.to_csv(feature_dir + f_name + '.csv', index=False, header=True)


if __name__ == '__main__':
    export_visual_feature(8)
    # export_visual_feature(16)
    # export_visual_feature(32)
    # export_visual_feature(64)
    # export_visual_feature(128)
