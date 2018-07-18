"""
拆分数据集
根据_flag_拆分为train、valid、test
"""

data_set_file = '../out/data_set.csv'
data_set_train = '../out/data_set.train'
data_set_test = '../out/data_set.test'
data_set_valid = '../out/data_set.valid'

if __name__ == '__main__':
    with open(data_set_file, 'r') as f_in, open(data_set_train, 'w') as f_train, open(data_set_valid, 'w') as f_valid, open(data_set_test, 'w') as f_test:
        for (num, line) in enumerate(f_in):
            if num == 0:
                f_train.write(line)
                f_valid.write(line)
                f_test.write(line)
            flag = line.split(',')[8]
            if flag == '0':
                f_train.write(line)
            elif flag == '1':
                f_valid.write(line)
            elif flag == '-1':
                f_test.write(line)