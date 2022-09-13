import os
# import argparse
import shutil

train_file_path = '/home/mona/codes/lama/datasets/afhq/train/train256_cat/train_shuffled_100.txt'

val_file_path = '/home/mona/codes/lama/datasets/afhq/train/train256_cat/val_shuffled.txt'

train_cat_all_path = '/home/mona/codes/lama/datasets/afhq/train_origin/cat'

names_txt_path = '/home/mona/codes/lama/datasets/afhq/test/all_cat_data_except_train100_val.txt'

target_path = '/home/mona/codes/lama/datasets/afhq/test/all_cat_data_except_train100_val'

# with open(train_file_path) as f:
#     train_names = [x.strip() for x in f.readlines()]
#     train_names.sort()

# with open(val_file_path) as f:
#     val_names = [x.strip() for x in f.readlines()]
#     val_names.sort()

# train100_val = list(set(train_names).union(set(val_names)))  # train_names + val_names
# print(len(train100_val))

# train_all =  os.listdir(train_cat_all_path)
# print(len(train_all))

# train_all_except_train100_val = list(set(train_all).difference(set(train100_val)))  # # train_all - train100_val
# print(len(train_all_except_train100_val))
# # with open(names_txt_path, 'w') as f:
# #     for name in train_all_except_train100_val:
# #         f.write(name+'\n')

# for name in train_all_except_train100_val:
#     shutil.copy(os.path.join(train_cat_all_path, name), os.path.join(target_path, name))



train_all =  os.listdir(target_path)
print(len(train_all))



