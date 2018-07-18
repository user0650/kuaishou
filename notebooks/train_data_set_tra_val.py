
"""
读取photo_ids和photo_visual_pca数据，放入内存
"""
photo_features = {}

file_photo_ids = '../out/train_photo.txt'
file_photo_features = '../out/train_photo_visual_pca.txt'
with open(file_photo_ids, 'r') as ids, open(file_photo_features, 'r') as features:
    for (num, line) in enumerate(ids):
        if (num + 1) % 10000 == 0:
            print(num + 1, flush=True)
        photo_id = line.replace("\n", "")
        photo_features[photo_id] = features.readline().replace("\n", "")


"""
读取train_interaction_ext_tra.txt，构造训练集
"""
file_in_tra = '../out/train_interaction_ext_tra.txt'
file_out_tra = '../out/train_data_set_tra.txt'
with open(file_in_tra, 'r') as f_in, open(file_out_tra, 'w') as f_out:
    for (num, line) in enumerate(f_in):
        if (num + 1) % 10000 == 0:
            print('tra: ', num + 1, flush=True)
        line = line.replace("\n", "")
        row = line.split("\t")
        label = row[2]
        user_id = row[0]
        photo_id = row[1]
        features = photo_features[photo_id]
        f_out.write(label + "\t" + user_id + "\t" + features + "\n")


"""
读取train_interaction_ext_val.txt，构造验证集
"""
file_in_val = '../out/train_interaction_ext_val.txt'
file_out_val = '../out/train_data_set_val.txt'
with open(file_in_val, 'r') as f_in, open(file_out_val, 'w') as f_out:
    for (num, line) in enumerate(f_in):
        if (num + 1) % 10000 == 0:
            print('val: ', num + 1, flush=True)
        line = line.replace("\n", "")
        row = line.split("\t")
        label = row[2]
        user_id = row[0]
        photo_id = row[1]
        features = photo_features[photo_id]
        f_out.write(label + "\t" + user_id + "\t" + features + "\n")
