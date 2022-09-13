import os
import shutil
# from random import sample
import random
from tqdm import tqdm




# train_all_path = '/home/mona/codes/lama/datasets/afhq/train_origin/wild'
# train_all_names = os.listdir(train_all_path)
# print(len(train_all_names))

# train_100 = sample(train_all_names, 100)
# print(len(train_100))

# train_rest = list(set(train_all_names).difference(set(train_100)))
# print(len(train_rest))

# val_500 = sample(train_rest, 500)
# print(len(val_500))

# train_100_target_path = '/home/mona/codes/lama/datasets/afhq/train/train256_wild/train256_wild_100'
# val_500_target_path = '/home/mona/codes/lama/datasets/afhq/train/train256_wild/val256_wild_500_source'
# train_100_txt = '/home/mona/codes/lama/datasets/afhq/train/train256_wild/train_shuffled_100.txt'
# val_500_txt = '/home/mona/codes/lama/datasets/afhq/train/train256_wild/val_shuffled_500.txt'

# with open(train_100_txt, 'w') as f:
#     for train_name in train_100:
#         f.write(train_name+'\n')
# for train_name in train_100:
#     shutil.copy(os.path.join(train_all_path, train_name), os.path.join(train_100_target_path, train_name))

# with open(val_500_txt, 'w') as f:
#     for val_name in val_500:
#         f.write(val_name+'\n')
# for val_name in val_500:
#     shutil.copy(os.path.join(train_all_path, val_name), os.path.join(val_500_target_path, val_name))


# train_all_path = '/home/mona/codes/lama/datasets/ukiyoe/ukiyoe-256/all'
# train_all_names = os.listdir(train_all_path)
# print(len(train_all_names))

# train_100 = sample(train_all_names, 100)
# print(len(train_100))

# train_rest = list(set(train_all_names).difference(set(train_100)))
# print(len(train_rest))

# val_500 = sample(train_rest, 500)
# print(len(val_500))

# train_rest_rest = list(set(train_rest).difference(set(val_500)))
# print(len(train_rest_rest))

# test_500 = sample(train_rest_rest, 500)
# print(len(test_500))

# train_100_target_path = '/home/mona/codes/lama/datasets/ukiyoe/ukiyoe-256/train256_ukiyoe_100'
# val_500_target_path = '/home/mona/codes/lama/datasets/ukiyoe/ukiyoe-256/val256_ukiyoe_500_source'
# test_500_target_path = '/home/mona/codes/lama/datasets/ukiyoe/ukiyoe-256/test256_ukiyoe_500_source'
# train_100_txt = '/home/mona/codes/lama/datasets/ukiyoe/ukiyoe-256/train_shuffled_100.txt'
# val_500_txt = '/home/mona/codes/lama/datasets/ukiyoe/ukiyoe-256/val_shuffled_500.txt'
# test_500_txt = '/home/mona/codes/lama/datasets/ukiyoe/ukiyoe-256/test_shuffled_500.txt'

# with open(train_100_txt, 'w') as f:
#     for train_name in train_100:
#         f.write(train_name+'\n')
# for train_name in train_100:
#     shutil.copy(os.path.join(train_all_path, train_name), os.path.join(train_100_target_path, train_name))

# with open(val_500_txt, 'w') as f:
#     for val_name in val_500:
#         f.write(val_name+'\n')
# for val_name in val_500:
#     shutil.copy(os.path.join(train_all_path, val_name), os.path.join(val_500_target_path, val_name))

# with open(test_500_txt, 'w') as f:
#     for test_name in test_500:
#         f.write(test_name+'\n')
# for test_name in test_500:
#     shutil.copy(os.path.join(train_all_path, test_name), os.path.join(test_500_target_path, test_name))

seed = 2

def seed_everything(seed): 
#  torch.manual_seed(seed) # Current CPU 
#  torch.cuda.manual_seed(seed) # Current GPU 
#  np.random.seed(seed) # Numpy module 
 random.seed(seed) # Python random module 
#  torch.backends.cudnn.benchmark = False # Close optimization 
#  torch.backends.cudnn.deterministic = True # Close optimization 
#  torch.cuda.manual_seed_all(seed) # All GPU (Optional) 

seed_everything(seed)

train_need_number = 100
val_number = 500

train_all_path = '/home/mona/codes/lama/datasets/afhq/train_origin/wild'
train_all_names = os.listdir(train_all_path)
print(len(train_all_names))

train_need = random.sample(train_all_names, train_need_number)
print(len(train_need))

train_rest = list(set(train_all_names).difference(set(train_need)))
print(len(train_rest))

val_500 = random.sample(train_rest, val_number)
print(len(val_500))

# train_rest = list(set(train_all_names).difference(set(val_500)))
# print(len(train_rest))
# --- set dir path and create ---
dataset_dir = '/home/mona/codes/lama/datasets/afhq/train/train256_wild/train256_wild_100_random_seed2'
train_need_target_path = os.path.join(dataset_dir, 'train')
val_500_target_path = os.path.join(dataset_dir, 'val256_cat_500_source')
val_path = os.path.join(dataset_dir, 'val')
train_need_txt = os.path.join(dataset_dir, 'train.txt')
val_500_txt = os.path.join(dataset_dir, 'val_shuffled_500.txt')

if not os.path.exists(train_need_target_path):
    os.makedirs(train_need_target_path)
if not os.path.exists(val_500_target_path):
    os.makedirs(val_500_target_path)
if not os.path.exists(val_path):
    os.makedirs(val_path)

with open(train_need_txt, 'w') as f:
    for train_name in train_need:
        f.write(train_name+'\n')
for train_name in tqdm(train_need):
    shutil.copy(os.path.join(train_all_path, train_name), os.path.join(train_need_target_path, train_name))

with open(val_500_txt, 'w') as f:
    for val_name in val_500:
        f.write(val_name+'\n')
for val_name in tqdm(val_500):
    shutil.copy(os.path.join(train_all_path, val_name), os.path.join(val_500_target_path, val_name))



seed = 3

def seed_everything(seed): 
#  torch.manual_seed(seed) # Current CPU 
#  torch.cuda.manual_seed(seed) # Current GPU 
#  np.random.seed(seed) # Numpy module 
 random.seed(seed) # Python random module 
#  torch.backends.cudnn.benchmark = False # Close optimization 
#  torch.backends.cudnn.deterministic = True # Close optimization 
#  torch.cuda.manual_seed_all(seed) # All GPU (Optional) 

seed_everything(seed)

train_need_number = 100
val_number = 500

train_all_path = '/home/mona/codes/lama/datasets/afhq/train_origin/wild'
train_all_names = os.listdir(train_all_path)
print(len(train_all_names))

train_need = random.sample(train_all_names, train_need_number)
print(len(train_need))

train_rest = list(set(train_all_names).difference(set(train_need)))
print(len(train_rest))

val_500 = random.sample(train_rest, val_number)
print(len(val_500))

# train_rest = list(set(train_all_names).difference(set(val_500)))
# print(len(train_rest))
# --- set dir path and create ---
dataset_dir = '/home/mona/codes/lama/datasets/afhq/train/train256_wild/train256_wild_100_random_seed3'
train_need_target_path = os.path.join(dataset_dir, 'train')
val_500_target_path = os.path.join(dataset_dir, 'val256_cat_500_source')
val_path = os.path.join(dataset_dir, 'val')
train_need_txt = os.path.join(dataset_dir, 'train.txt')
val_500_txt = os.path.join(dataset_dir, 'val_shuffled_500.txt')

if not os.path.exists(train_need_target_path):
    os.makedirs(train_need_target_path)
if not os.path.exists(val_500_target_path):
    os.makedirs(val_500_target_path)
if not os.path.exists(val_path):
    os.makedirs(val_path)

with open(train_need_txt, 'w') as f:
    for train_name in train_need:
        f.write(train_name+'\n')
for train_name in tqdm(train_need):
    shutil.copy(os.path.join(train_all_path, train_name), os.path.join(train_need_target_path, train_name))

with open(val_500_txt, 'w') as f:
    for val_name in val_500:
        f.write(val_name+'\n')
for val_name in tqdm(val_500):
    shutil.copy(os.path.join(train_all_path, val_name), os.path.join(val_500_target_path, val_name))