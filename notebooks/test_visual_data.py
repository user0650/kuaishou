import numpy as np
import time


def get_test_visual_data(photo_id, n=10):
    """
    读取一个作品的视觉特征
    取TopN特征值的索引
    """
    v = np.load('../../data/preliminary_visual_test/' + str(photo_id))
    return v.argsort()[:, :-(n + 1):-1]


def get_test_visual_str(photo_id, n):
    arr = get_test_visual_data(photo_id, n)[0]
    return "\t".join(str(x) for x in arr)


def gen(file_in, file_out, offset=0, count=100, feature_count=10):
    """
    读取交互数据，拼接photo_id对应的视觉特征数据
    """
    with open(file_in, mode='r') as f_in, open(file_out, mode='w') as f_out:
        counter = 0
        for (num, line) in enumerate(f_in):
            if num < offset:
                continue

            line = line.replace("\n", "")
            fields = line.split("\t")
            photo_feature = get_test_visual_str(fields[1], feature_count)
            f_out.write(fields[4] + "\t" + fields[5] + "\t" + photo_feature + "\n")

            if counter % 10000 == 0:
                print(time.time(), " -> ", counter, flush=True)
            counter = counter + 1

            if 0 < count <= counter:
                break


if __name__ == '__main__':
    gen('../out/test_interaction_ext.txt', '../out/test_visual_100.txt', 0, 0, 100)

