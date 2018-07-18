# view_cn、click_cn、like_cn、follow_cn
# view_cn_0、click_cn_0、like_cn_0、follow_cn_0 (type=0，图片)
# view_cn_1、click_cn_1、like_cn_1、follow_cn_1 (type=1，短视频)
# ...
# view_cn_5、click_cn_5、like_cn_5、follow_cn_5 (type=5，长视频)
#
# -- 4*7 = 28
#
# click_cn/view_cn、like_cn/view_cn、follow_cn/view_cn、like_cn/click_cn、follow_cn/click_cn、follow_cn/like_cn
# click_cn_0/view_cn_0、like_cn_0/view_cn_0、follow_cn_0/view_cn_0、like_cn_0/click_cn_0、follow_cn_0/click_cn_0、follow_cn_0/like_cn_0
# ...
# click_cn_5/view_cn_5、like_cn_5/view_cn_5、follow_cn_5/view_cn_5、like_cn_5/click_cn_5、follow_cn_5/click_cn_5、follow_cn_5/like_cn_5
#
# -- 6*7 = 42
#
# 以下均为video统计特征，不含picture
# duration_time_view_sum
# duration_time_view_avg
# duration_time_view_mode
# duration_time_view_max
# duration_time_view_min
#
# playing_time_click_sum、duration_time_click_sum
# playing_time_click_avg、duration_tiem_click_avg
# playing_time_click_mode、duration_time_click_mode
# playing_time_click_max、duration_time_click_max
# playing_time_click_min、duration_time_click_min
#
# playing_time_like_sum、duration_time_like_sum
# playing_time_like_avg、duration_tiem_like_avg
# playing_time_like_mode、duration_time_like_mode
# playing_time_like_max、duration_time_like_max
# playing_time_like_min、duration_time_like_min
#
# playing_time_follow_sum、duration_time_follow_sum
# playing_time_follow_avg、duration_tiem_follow_avg
# playing_time_follow_mode、duration_time_follow_mode
# playing_time_follow_max、duration_time_follow_max
# playing_time_follow_min、duration_time_follow_min
#
# -- 10*3 + 5 = 35

# 共 105 个特征

import pandas as pd
import numpy as np
import os

in_txt = '../out/base_feature.csv'
out_txt = '../out/stat_feature.csv'
out_dir = '../out/'


def export_stat_feature(df, f_name, stats, default_value=0):
    ls = []
    for user_id in df['user_id']:
        if user_id in stats:
            ls.append(stats[user_id])
        else:
            ls.append(default_value)
    print(f_name)
    pd.DataFrame({f_name: ls}).to_csv(out_dir + f_name + '.csv', index=False, header=True)


# view_cn
def export_view_cn_feature(df):
    f_name = 'f_view_cn'
    stats = df.groupby('user_id')['photo_id'].count()
    export_stat_feature(df, f_name, stats)


# click_cn
def export_click_cn_feature(df):
    f_name = 'f_click_cn'
    stats = df.groupby('user_id')['click'].sum()  # click=0,1
    export_stat_feature(df, f_name, stats)


# like_cn
def export_like_cn_feature(df):
    f_name = 'f_like_cn'
    stats = df.groupby('user_id')['like'].sum()  # like=0,1
    export_stat_feature(df, f_name, stats)


# follow_cn
def export_follow_cn_feature(df):
    f_name = 'f_follow_cn'
    stats = df.groupby('user_id')['follow'].sum()  # follow=0,1
    export_stat_feature(df, f_name, stats)


# view_cn_t
# t - photo_type,0~5
def export_view_cn_t_feature(df, t):
    f_name = 'f_view_cn_' + str(t)
    stats = df[df['f_photo_type'] == t].groupby('user_id')['photo_id'].count()
    export_stat_feature(df, f_name, stats)


# click_cn_0
def export_click_cn_t_feature(df, t):
    f_name = 'f_click_cn_' + str(t)
    stats = df[df['f_photo_type'] == t].groupby('user_id')['click'].sum()
    export_stat_feature(df, f_name, stats)


# like_cn_0
def export_like_cn_t_feature(df, t):
    f_name = 'f_like_cn_' + str(t)
    stats = df[df['f_photo_type'] == t].groupby('user_id')['like'].sum()
    export_stat_feature(df, f_name, stats)


# follow_cn_0
def export_follow_cn_t_feature(df, t):
    f_name = 'f_follow_cn_' + str(t)
    stats = df[df['f_photo_type'] == t].groupby('user_id')['follow'].sum()
    export_stat_feature(df, f_name, stats)


# click_cn/view_cn、like_cn/view_cn、follow_cn/view_cn、like_cn/click_cn、follow_cn/click_cn、follow_cn/like_cn
def export_rate_feature(f_name, f1_name, f2_name):
    df1 = pd.read_csv(out_dir + f1_name + '.csv')
    df2 = pd.read_csv(out_dir + f2_name + '.csv')
    arr = (df1[f1_name] / df2[f2_name]).fillna(0.).replace(np.inf, 0.).values
    pd.DataFrame({f_name: arr}).to_csv(out_dir + f_name + '.csv', index=False, header=True)


def export_click_view_rate_feature():
    export_rate_feature('f_click_view_rate', 'f_click_cn', 'f_view_cn')


def export_like_view_rate_feature():
    export_rate_feature('f_like_view_rate', 'f_like_cn', 'f_view_cn')


def export_follow_view_rate_feature():
    export_rate_feature('f_follow_view_rate', 'f_follow_cn', 'f_view_cn')


def export_like_click_rate_feature():
    export_rate_feature('f_like_click_rate', 'f_like_cn', 'f_click_cn')


def export_follow_click_rate_feature():
    export_rate_feature('f_follow_click_rate', 'f_follow_cn', 'f_click_cn')


def export_follow_like_rate_feature():
    export_rate_feature('f_follow_like_rate', 'f_follow_cn', 'f_like_cn')


def export_click_view_rate_t_feature(t):
    export_rate_feature('f_click_view_rate_' + str(t), 'f_click_cn_' + str(t), 'f_view_cn_' + str(t))


def export_like_view_rate_t_feature(t):
    export_rate_feature('f_like_view_rate_' + str(t), 'f_like_cn_' + str(t), 'f_view_cn_' + str(t))


def export_follow_view_rate_t_feature(t):
    export_rate_feature('f_follow_view_rate_' + str(t), 'f_follow_cn_' + str(t), 'f_view_cn_' + str(t))


def export_like_click_rate_t_feature(t):
    export_rate_feature('f_like_click_rate_' + str(t), 'f_like_cn_' + str(t), 'f_click_cn_' + str(t))


def export_follow_click_rate_t_feature(t):
    export_rate_feature('f_follow_click_rate_' + str(t), 'f_follow_cn_' + str(t), 'f_click_cn_' + str(t))


def export_follow_like_rate_t_feature(t):
    export_rate_feature('f_follow_like_rate_' + str(t), 'f_follow_cn_' + str(t), 'f_like_cn_' + str(t))


# 浏览的作品的总时长（photo_type > 0）
# duration_time_view_sum
def export_duration_time_view_sum_feature(df):
    f_name = 'f_duration_time_view_sum'
    stats = df[df['f_photo_type'] > 0].groupby('user_id')['duration_time'].sum()
    export_stat_feature(df, f_name, stats)


# duration_tiem_view_mean
def export_duration_time_view_mean_feature(df):
    f_name = 'f_duration_time_view_mean'
    stats = df[df['f_photo_type'] > 0].groupby('user_id')['duration_time'].mean()
    export_stat_feature(df, f_name, stats)


# duration_time_view_mid
def export_duration_time_view_mid_feature(df):
    f_name = 'f_duration_time_view_mid'
    stats = df[df['f_photo_type'] > 0].groupby('user_id')['duration_time'].median()
    export_stat_feature(df, f_name, stats)


# duration_time_view_max
def export_duration_time_view_max_feature(df):
    f_name = 'f_duration_time_view_max'
    stats = df[df['f_photo_type'] > 0].groupby('user_id')['duration_time'].max()
    export_stat_feature(df, f_name, stats)


# duration_time_view_min
def export_duration_time_view_min_feature(df):
    f_name = 'f_duration_time_view_min'
    stats = df[df['f_photo_type'] > 0].groupby('user_id')['duration_time'].min()
    export_stat_feature(df, f_name, stats)


def export_playing_time_click_sum_feature(df):
    f_name = 'f_playing_time_click_sum'
    stats = df[(df['f_photo_type'] > 0) & (df['click'] == 1)].groupby('user_id')['playing_time'].sum()
    export_stat_feature(df, f_name, stats)


def export_playing_time_click_mean_feature(df):
    f_name = 'f_playing_time_click_mean'
    stats = df[(df['f_photo_type'] > 0) & (df['click'] == 1)].groupby('user_id')['playing_time'].mean()
    export_stat_feature(df, f_name, stats)


def export_playing_time_click_mid_feature(df):
    f_name = 'f_playing_time_click_mid'
    stats = df[(df['f_photo_type'] > 0) & (df['click'] == 1)].groupby('user_id')['playing_time'].median()
    export_stat_feature(df, f_name, stats)


def export_playing_time_click_max_feature(df):
    f_name = 'f_playing_time_click_max'
    stats = df[(df['f_photo_type'] > 0) & (df['click'] == 1)].groupby('user_id')['playing_time'].max()
    export_stat_feature(df, f_name, stats)


def export_playing_time_click_min_feature(df):
    f_name = 'f_playing_time_click_min'
    stats = df[(df['f_photo_type'] > 0) & (df['click'] == 1)].groupby('user_id')['playing_time'].min()
    export_stat_feature(df, f_name, stats)


def export_duration_time_click_sum_feature(df):
    f_name = 'f_duration_time_click_sum'
    stats = df[(df['f_photo_type'] > 0) & (df['click'] == 1)].groupby('user_id')['duration_time'].sum()
    export_stat_feature(df, f_name, stats)


def export_duration_time_click_mean_feature(df):
    f_name = 'f_duration_time_click_mean'
    stats = df[(df['f_photo_type'] > 0) & (df['click'] == 1)].groupby('user_id')['duration_time'].mean()
    export_stat_feature(df, f_name, stats)


def export_duration_time_click_mid_feature(df):
    f_name = 'f_duration_time_click_mid'
    stats = df[(df['f_photo_type'] > 0) & (df['click'] == 1)].groupby('user_id')['duration_time'].median()
    export_stat_feature(df, f_name, stats)


def export_duration_time_click_max_feature(df):
    f_name = 'f_duration_time_click_max'
    stats = df[(df['f_photo_type'] > 0) & (df['click'] == 1)].groupby('user_id')['duration_time'].max()
    export_stat_feature(df, f_name, stats)


def export_duration_time_click_min_feature(df):
    f_name = 'f_duration_time_click_min'
    stats = df[(df['f_photo_type'] > 0) & (df['click'] == 1)].groupby('user_id')['duration_time'].min()
    export_stat_feature(df, f_name, stats)


def export_playing_time_like_sum_feature(df):
    f_name = 'f_playing_time_like_sum'
    stats = df[(df['f_photo_type'] > 0) & (df['like'] == 1)].groupby('user_id')['playing_time'].sum()
    export_stat_feature(df, f_name, stats)


def export_playing_time_like_mean_feature(df):
    f_name = 'f_playing_time_like_mean'
    stats = df[(df['f_photo_type'] > 0) & (df['like'] == 1)].groupby('user_id')['playing_time'].mean()
    export_stat_feature(df, f_name, stats)


def export_playing_time_like_mid_feature(df):
    f_name = 'f_playing_time_like_mid'
    stats = df[(df['f_photo_type'] > 0) & (df['like'] == 1)].groupby('user_id')['playing_time'].median()
    export_stat_feature(df, f_name, stats)


def export_playing_time_like_max_feature(df):
    f_name = 'f_playing_time_like_max'
    stats = df[(df['f_photo_type'] > 0) & (df['like'] == 1)].groupby('user_id')['playing_time'].max()
    export_stat_feature(df, f_name, stats)


def export_playing_time_like_min_feature(df):
    f_name = 'f_playing_time_like_min'
    stats = df[(df['f_photo_type'] > 0) & (df['like'] == 1)].groupby('user_id')['playing_time'].min()
    export_stat_feature(df, f_name, stats)


def export_duration_time_like_sum_feature(df):
    f_name = 'f_duration_time_like_sum'
    stats = df[(df['f_photo_type'] > 0) & (df['like'] == 1)].groupby('user_id')['duration_time'].sum()
    export_stat_feature(df, f_name, stats)


def export_duration_time_like_mean_feature(df):
    f_name = 'f_duration_time_like_mean'
    stats = df[(df['f_photo_type'] > 0) & (df['like'] == 1)].groupby('user_id')['duration_time'].mean()
    export_stat_feature(df, f_name, stats)


def export_duration_time_like_mid_feature(df):
    f_name = 'f_duration_time_like_mid'
    stats = df[(df['f_photo_type'] > 0) & (df['like'] == 1)].groupby('user_id')['duration_time'].median()
    export_stat_feature(df, f_name, stats)


def export_duration_time_like_max_feature(df):
    f_name = 'f_duration_time_like_max'
    stats = df[(df['f_photo_type'] > 0) & (df['like'] == 1)].groupby('user_id')['duration_time'].max()
    export_stat_feature(df, f_name, stats)


def export_duration_time_like_min_feature(df):
    f_name = 'f_duration_time_like_min'
    stats = df[(df['f_photo_type'] > 0) & (df['like'] == 1)].groupby('user_id')['duration_time'].min()
    export_stat_feature(df, f_name, stats)


def export_playing_time_follow_sum_feature(df):
    f_name = 'f_playing_time_follow_sum'
    stats = df[(df['f_photo_type'] > 0) & (df['follow'] == 1)].groupby('user_id')['playing_time'].sum()
    export_stat_feature(df, f_name, stats)


def export_playing_time_follow_mean_feature(df):
    f_name = 'f_playing_time_follow_mean'
    stats = df[(df['f_photo_type'] > 0) & (df['follow'] == 1)].groupby('user_id')['playing_time'].mean()
    export_stat_feature(df, f_name, stats)


def export_playing_time_follow_mid_feature(df):
    f_name = 'f_playing_time_follow_mid'
    stats = df[(df['f_photo_type'] > 0) & (df['follow'] == 1)].groupby('user_id')['playing_time'].median()
    export_stat_feature(df, f_name, stats)


def export_playing_time_follow_max_feature(df):
    f_name = 'f_playing_time_follow_max'
    stats = df[(df['f_photo_type'] > 0) & (df['follow'] == 1)].groupby('user_id')['playing_time'].max()
    export_stat_feature(df, f_name, stats)


def export_playing_time_follow_min_feature(df):
    f_name = 'f_playing_time_follow_min'
    stats = df[(df['f_photo_type'] > 0) & (df['follow'] == 1)].groupby('user_id')['playing_time'].min()
    export_stat_feature(df, f_name, stats)


def export_duration_time_follow_sum_feature(df):
    f_name = 'f_duration_time_follow_sum'
    stats = df[(df['f_photo_type'] > 0) & (df['follow'] == 1)].groupby('user_id')['duration_time'].sum()
    export_stat_feature(df, f_name, stats)


def export_duration_time_follow_mean_feature(df):
    f_name = 'f_duration_time_follow_mean'
    stats = df[(df['f_photo_type'] > 0) & (df['follow'] == 1)].groupby('user_id')['duration_time'].mean()
    export_stat_feature(df, f_name, stats)


def export_duration_time_follow_mid_feature(df):
    f_name = 'f_duration_time_follow_mid'
    stats = df[(df['f_photo_type'] > 0) & (df['follow'] == 1)].groupby('user_id')['duration_time'].median()
    export_stat_feature(df, f_name, stats)


def export_duration_time_follow_max_feature(df):
    f_name = 'f_duration_time_follow_max'
    stats = df[(df['f_photo_type'] > 0) & (df['follow'] == 1)].groupby('user_id')['duration_time'].max()
    export_stat_feature(df, f_name, stats)


def export_duration_time_follow_min_feature(df):
    f_name = 'f_duration_time_follow_min'
    stats = df[(df['f_photo_type'] > 0) & (df['follow'] == 1)].groupby('user_id')['duration_time'].min()
    export_stat_feature(df, f_name, stats)


def export_features():
    df_interaction = pd.read_csv(in_txt)

    export_view_cn_feature(df_interaction)
    export_click_cn_feature(df_interaction)
    export_like_cn_feature(df_interaction)
    export_follow_cn_feature(df_interaction)

    for t in range(6):
        export_view_cn_t_feature(df_interaction, t)
        export_click_cn_t_feature(df_interaction, t)
        export_like_cn_t_feature(df_interaction, t)
        export_follow_cn_t_feature(df_interaction, t)

    export_duration_time_view_sum_feature(df_interaction)
    export_duration_time_view_mean_feature(df_interaction)
    export_duration_time_view_mid_feature(df_interaction)
    export_duration_time_view_max_feature(df_interaction)
    export_duration_time_view_min_feature(df_interaction)
    export_playing_time_click_sum_feature(df_interaction)
    export_playing_time_click_mean_feature(df_interaction)
    export_playing_time_click_mid_feature(df_interaction)
    export_playing_time_click_max_feature(df_interaction)
    export_playing_time_click_min_feature(df_interaction)
    export_duration_time_click_sum_feature(df_interaction)
    export_duration_time_click_mean_feature(df_interaction)
    export_duration_time_click_mid_feature(df_interaction)
    export_duration_time_click_max_feature(df_interaction)
    export_duration_time_click_min_feature(df_interaction)
    export_playing_time_like_sum_feature(df_interaction)
    export_playing_time_like_mean_feature(df_interaction)
    export_playing_time_like_mid_feature(df_interaction)
    export_playing_time_like_max_feature(df_interaction)
    export_playing_time_like_min_feature(df_interaction)
    export_duration_time_like_sum_feature(df_interaction)
    export_duration_time_like_mean_feature(df_interaction)
    export_duration_time_like_mid_feature(df_interaction)
    export_duration_time_like_max_feature(df_interaction)
    export_duration_time_like_min_feature(df_interaction)
    export_playing_time_follow_sum_feature(df_interaction)
    export_playing_time_follow_mean_feature(df_interaction)
    export_playing_time_follow_mid_feature(df_interaction)
    export_playing_time_follow_max_feature(df_interaction)
    export_playing_time_follow_min_feature(df_interaction)
    export_duration_time_follow_sum_feature(df_interaction)
    export_duration_time_follow_mean_feature(df_interaction)
    export_duration_time_follow_mid_feature(df_interaction)
    export_duration_time_follow_max_feature(df_interaction)
    export_duration_time_follow_min_feature(df_interaction)

    export_click_view_rate_feature()
    export_like_view_rate_feature()
    export_follow_view_rate_feature()
    export_like_click_rate_feature()
    export_follow_click_rate_feature()
    export_follow_like_rate_feature()

    for t in range(6):
        export_click_view_rate_t_feature(t)
        export_like_view_rate_t_feature(t)
        export_follow_view_rate_t_feature(t)
        export_like_click_rate_t_feature(t)
        export_follow_click_rate_t_feature(t)
        export_follow_like_rate_t_feature(t)


def paste_features():
    os.system("paste -d',' " + in_txt + " ../out/f_*.csv > " + out_txt + "")
    os.system("rm -rf " + out_dir + "/f_*.csv")


if __name__ == '__main__':
    export_features()
    paste_features()
