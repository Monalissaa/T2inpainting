import os

test_origin_path = '/home/mona/codes/lama/datasets/afhq/test_origin/wild'

test_thick_path = '/home/mona/codes/lama/datasets/afhq/test/test_256_with_mask/wild/random_thick_256'

list_test_origin = os.listdir(test_origin_path)
list_test_thick = [x[:-12]+'.jpg' for x in os.listdir(test_thick_path) if 'mask' not in x]

for x in list_test_origin:
    if x not in list_test_thick:
        print(x)