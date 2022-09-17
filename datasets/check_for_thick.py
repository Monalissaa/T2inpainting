import os

test_origin_path = '/home/mona/codes/lama/datasets/MetFace/test_500_source_random_seed10002'

test_thick_path = '/home/mona/codes/lama/datasets/MetFace/test_500_with_mask_random_seed10002/random_thick_256'

list_test_origin = os.listdir(test_origin_path)
list_test_thick = [x[:-12]+'.png' for x in os.listdir(test_thick_path) if 'mask' not in x]

for x in list_test_origin:
    if x not in list_test_thick:
        print(x)