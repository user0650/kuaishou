"""
计算样本权重

样本权重由三部分构成：like、follow、playing_time/duration_time，权重值为0~1的浮点数

三部分所占比例用线性回归进行拟合
"""

import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import SGDRegressor

in_txt = "../out/base_feature.csv"
out_txt = "../out/weight.csv"
model = "../model/model_sample_weight"
batch_size = 8192


def fit_weight():
    with open(in_txt, 'r') as f_in:
        X_batch = []
        y_batch = []
        reg = SGDRegressor()
        for (num, line) in enumerate(f_in):
            if num == 0:
                continue
            line = line.replace("\n", "")
            row = [float(x) for x in line.split(',')]
            click = row[2]
            like = row[3]
            follow = row[4]
            playing_time = row[6]
            duration_time = row[7]
            flag = row[8]
            if flag < 0:  # test数据不参与训练
                continue
            play_cn = min(playing_time / duration_time, 1.0) if duration_time > 0 else click
            X_batch.append([like, follow, play_cn])
            y_batch.append(click)
            if (num + 1) % batch_size == 0:
                print('fit: ', num + 1, flush=True)
                reg.partial_fit(X_batch, y_batch)
                X_batch.clear()
                y_batch.clear()
        reg.partial_fit(X_batch, y_batch)
        X_batch.clear()
        y_batch.clear()
        joblib.dump(reg, model)


def predict_weight():
    with open(in_txt, 'r') as f_in:
        X_batch = []
        reg = joblib.load(model)
        weights = []
        for (num, line) in enumerate(f_in):
            if num == 0:
                continue
            line = line.replace("\n", "")
            row = [float(x) for x in line.split(',')]
            click = row[2]
            like = row[3]
            follow = row[4]
            playing_time = row[6]
            duration_time = row[7]
            play_cn = min(playing_time / duration_time, 1.0) if duration_time > 0 else click
            X_batch.append([like, follow, play_cn])
            if (num + 1) % batch_size == 0:
                print("predict: ", num + 1, flush=True)
                for w in reg.predict(X_batch):
                    weights.append(min(w, 1.0))
                # print(len(weights))
                X_batch.clear()
        for w in reg.predict(X_batch):
            weights.append(min(w, 1.0))
        X_batch.clear()
        pd.DataFrame(data={'weight': weights}).to_csv(out_txt, index=False, header=True)


if __name__ == '__main__':
    fit_weight()
    reg = joblib.load(model)
    print(reg.coef_)
    print(reg.intercept_)
    predict_weight()

    # cn = 52134536
    # no_click_cn = 41587484
    # click_cn = 10547052
    #
    # no_click_weight = 1 - no_click_cn / cn
    # click_weight = 1 - click_cn / cn
    #
    # no_like_follow_cn = 10351065
    # like_follow_cn = 195987
    # no_like_follow_weight = click_weight * (1 - no_like_follow_cn / click_cn)
    # like_follow_weight = click_weight * (1 - like_follow_cn / click_cn)
    #
    # play_weight = no_like_follow_weight
    #
    # like_cn = 152349
    # follow_cn = 51344
    #
    # like_weight = click_weight * 1 * (1 - like_cn / like_follow_cn)
    # follow_weight = click_weight * 1 * (1 - follow_cn / like_follow_cn)
    #
    # print(no_click_weight, like_weight, follow_weight, play_weight)
    # print(no_click_weight + like_weight + follow_weight + play_weight)
    #
    # with open(in_txt, 'r') as f_in:
    #     weights = []
    #     for (num, line) in enumerate(f_in):
    #         line = line.replace("\n", "")
    #         row = [float(x) for x in line.split('\t')]
    #         click = row[2]
    #         like = row[3]
    #         follow = row[4]
    #         playing_time = row[6]
    #         duration_time = row[7]
    #         play_cn = min(playing_time / duration_time, 1.0) if duration_time > 0 else click
    #         weights.append(no_click_weight + like_weight * like + follow_weight * follow + play_weight * play_cn)
    #         if (num + 1) % batch_size == 0:
    #             print("weight: ", num + 1, flush=True)
    #     pd.DataFrame(data={'weight': weights}).to_csv(out_txt, index=False, header=True)
