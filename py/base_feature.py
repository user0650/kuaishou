# time => hour、minute
# duration_time => photo_type(图片、短视频<10、中短视频10~30、中视频30~60、中长视频60~120、长视频>120)

import pandas as pd
import os

# train_interaction_txt = '/gdata/Kwai2018Final/train/train_interaction.txt'
# test_interaction_txt = '/gdata/Kwai2018Final/test/test_interaction.txt'
train_interaction_txt = '../final_contest/train/train_interaction_10000.txt'
test_interaction_txt = '../final_contest/test/test_interaction_1000.txt'

out_txt = '../out/base_feature.csv'
out_dir = '../out/'


# load interaction_data
def load_interaction_data():
    columns = ['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time', '_flag_']
    df_train = pd.read_table(train_interaction_txt, names=columns)

    df_tmp = df_train.sort_values('time')
    df_tmp_tra = df_tmp.iloc[:int(df_tmp.shape[0] * 0.8), :].copy()
    df_tmp_val = df_tmp.iloc[int(df_tmp.shape[0] * 0.8):, :].copy()
    valid_photo_set = set(df_tmp_val['photo_id']) - set(df_tmp_tra['photo_id'])
    df_train['_flag_'] = df_train.apply(lambda row: 1 if row['photo_id'] in valid_photo_set else 0, axis=1)

    df_test = pd.read_table(test_interaction_txt, names=['user_id', 'photo_id', 'time', 'duration_time'])
    df_test = df_test.reindex(columns=columns)
    # fillna
    df_test['click'] = 0
    df_test['like'] = 0
    df_test['follow'] = 0
    df_test['playing_time'] = 0
    df_test['_flag_'] = -1  # test data, _flag_=-1
    return pd.concat([df_train, df_test], axis=0)


# time => hour，0~23
def export_hour_feature(df):
    time_series = df['time']
    time_min = time_series.min()
    ls = []
    for time in time_series:
        ls.append(int(round((time - time_min) / 1000 / 60 / 60 % 24)))
    f_name = 'f_hour'
    pd.DataFrame({f_name: ls}).to_csv(out_dir + f_name + '.csv', index=False, header=True)


# time => minute，0~1440
def export_minute_feature(df):
    time_series = df['time']
    time_min = time_series.min()
    ls = []
    for time in time_series:
        ls.append(int(round((time - time_min) / 1000 / 60 % (24 * 60))))
    f_name = 'f_minute'
    pd.DataFrame({f_name: ls}).to_csv(out_dir + f_name + '.csv', index=False, header=True)


# duration_time => photo_type(图片、短视频<10、中短视频10~30、中视频30~60、中长视频60~120、长视频>120)
def export_photo_type_feature(df):
    duration_time_series = df['duration_time']
    ls = []
    for duration_time in duration_time_series:
        if duration_time == 0:
            photo_type = 0
        elif 1 <= duration_time < 10:
            photo_type = 1
        elif 10 <= duration_time < 30:
            photo_type = 2
        elif 30 <= duration_time < 60:
            photo_type = 3
        elif 60 <= duration_time < 120:
            photo_type = 4
        else:
            photo_type = 5
        ls.append(photo_type)
    f_name = 'f_photo_type'
    pd.DataFrame({f_name: ls}).to_csv(out_dir + f_name + '.csv', index=False, header=True)


def paste_features():
    os.system("paste -d',' ../out/tmp.csv ../out/f_hour.csv ../out/f_minute.csv ../out/f_photo_type.csv > " + out_txt)
    os.system("rm -rf ../out/tmp.csv ../out/f_*.csv")


if __name__ == '__main__':
    df_interaction = load_interaction_data()
    df_interaction.to_csv('../out/tmp.csv', header=True, index=False)

    export_hour_feature(df_interaction)
    export_minute_feature(df_interaction)
    export_photo_type_feature(df_interaction)

    paste_features()
