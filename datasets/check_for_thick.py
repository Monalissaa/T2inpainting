import os

test_origin_path = '/home/mona/codes/lama/datasets/afhq/test_origin/ukiyoe'

test_thick_path = '/home/mona/codes/lama/datasets/ukiyoe/ukiyoe-256/test256_ukiyoe_500_with_mask/random_thick_256'

list_test_origin = os.listdir(test_origin_path)
list_test_thick = [x[:-12]+'.jpg' for x in os.listdir(test_thick_path) if 'mask' not in x]

for x in list_test_origin:
    if x not in list_test_thick:
        print(x)