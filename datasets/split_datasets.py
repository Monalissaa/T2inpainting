import os
# import argparse
import shutil
from tqdm import tqdm

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



# train_all =  os.listdir(target_path)
# print(len(train_all))


all_img_path = '/home/szh/ZYB/lsun/lsun/church_outdoor_all/all'
test_img_path = '/home/szh/ZYB/lsun/lsun/church_outdoor_all/test-2000-source-seed1002'
all_except_test_path = '/home/szh/ZYB/lsun/lsun/church_outdoor_all/all-except-test'

# all_img_filename = os.listdir(all_img_path)
# test_img_filename = os.listdir(test_img_path)

# all_except_test = list(set(all_img_filename).difference(set(test_img_filename)))

# for filename in tqdm(all_except_test):
#     shutil.copy(os.path.join(all_img_path, filename), os.path.join(all_except_test_path, filename))

# print(len(all_img_filename))
# print(len(test_img_filename))
# print(len(all_except_test))

# all_except_test_list = os.listdir(all_except_test_path)
# print(len(all_except_test_list))

import random
seed = 10002

def seed_everything(seed): 
#  torch.manual_seed(seed) # Current CPU 
#  torch.cuda.manual_seed(seed) # Current GPU 
#  np.random.seed(seed) # Numpy module 
 random.seed(seed) # Python random module 
#  torch.backends.cudnn.benchmark = False # Close optimization 
#  torch.backends.cudnn.deterministic = True # Close optimization 
#  torch.cuda.manual_seed_all(seed) # All GPU (Optional) 

seed_everything(seed)

all_img_filename = os.listdir(all_img_path)
print(len(all_img_filename))

test_need = random.sample(all_img_filename, 2000)
print(len(test_need))

all_except_test = list(set(all_img_filename).difference(set(test_need)))
print(len(all_except_test))

for test_name in tqdm(test_need):
    shutil.copy(os.path.join(all_img_path, test_name), os.path.join(test_img_path, test_name))

for filename in tqdm(all_except_test):
    shutil.copy(os.path.join(all_img_path, filename), os.path.join(all_except_test_path, filename))
