
"""
data processing
"""

import pandas as pd
import numpy as np

train_interaction_txt = '/gdata/Kwai2018Final/train/train_interaction.txt'
test_interaction_txt = '/gdata/Kwai2018Final/test/test_interaction.txt'
train_visual_dir = '/gdata/Kwai2018Final/train/final_visual_train/'
test_visual_dir = '/gdata/Kwai2018Final/test/final_visual_test/'

photo_visual_txt = '../out/photo_visual.txt'


def photo_visual_data():
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


if __name__ == '__main__':
    photo_visual_data()
