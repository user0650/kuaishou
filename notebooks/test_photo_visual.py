import pandas as pd
import numpy as np


def get_test_visual_data(photo_id):
    """
    读取一个作品的视觉特征
    取TopN特征值的索引
    """
    return np.load('../../data/preliminary_visual_test/' + photo_id)


def get_test_visual_str(photo_id):
    arr = get_test_visual_data(photo_id)[0]
    return "\t".join(str(x) for x in arr)


def gen_feature_text():
    """
    把所有photo视觉信息合并成文件
    id  f1  f2  f3  ... f2048
    """
    with open('../out/test_photo.txt', 'r') as file_in, open('../out/b.txt', 'w') as file_out:
        for (num, line) in enumerate(file_in):
            if num <= 20000:
                continue
            photo_id = line.replace("\n", "")
            features = get_test_visual_str(photo_id)
            file_out.write(features + "\n")
            if num >= 30000:
                break


if __name__ == '__main__':
    gen_feature_text()
