#python3 bin/gen_mask_dataset.py \
#$(pwd)/configs/data_gen/random_thick_256.yaml \
#datasets/celeba-hq-dataset/val_source_256/ \
#datasets/celeba-hq-dataset/val_256/random_thick_256/
#
#python3 bin/gen_mask_dataset.py \
#$(pwd)/configs/data_gen/random_thin_256.yaml \
#datasets/celeba-hq-dataset/val_source_256/ \
#datasets/celeba-hq-dataset/val_256/random_thin_256/
#
#python3 bin/gen_mask_dataset.py \
#$(pwd)/configs/data_gen/random_medium_256.yaml \
#datasets/celeba-hq-dataset/val_source_256/ \
#datasets/celeba-hq-dataset/val_256/random_medium_256/
#
#python3 bin/gen_mask_dataset.py \
#$(pwd)/configs/data_gen/random_thick_256.yaml \
#datasets/celeba-hq-dataset/visual_test_source_256/ \
#datasets/celeba-hq-dataset/visual_test_256/random_thick_256/
#
#python3 bin/gen_mask_dataset.py \
#$(pwd)/configs/data_gen/random_thin_256.yaml \
#datasets/celeba-hq-dataset/visual_test_source_256/ \
#datasets/celeba-hq-dataset/visual_test_256/random_thin_256/
#
#python3 bin/gen_mask_dataset.py \
#$(pwd)/configs/data_gen/random_medium_256.yaml \
#datasets/celeba-hq-dataset/visual_test_source_256/ \
#datasets/celeba-hq-dataset/visual_test_256/random_medium_256/


#python3 bin/gen_mask_dataset.py \
#$(pwd)/configs/data_gen/random_thick_256.yaml \
#datasets/afhq/train/train256_cat/val256_cat_500_source/ \
#datasets/afhq/train/train256_cat/val256_cat_500/random_thick_256/
#
#python3 bin/gen_mask_dataset.py \
#$(pwd)/configs/data_gen/random_thin_256.yaml \
#datasets/afhq/train/train256_cat/val256_cat_500_source/ \
#datasets/afhq/train/train256_cat/val256_cat_500/random_thin_256/
#
#python3 bin/gen_mask_dataset.py \
#$(pwd)/configs/data_gen/random_medium_256.yaml \
#datasets/afhq/train/train256_cat/val256_cat_500_source/ \
#datasets/afhq/train/train256_cat/val256_cat_500/random_medium_256/

# python3 bin/gen_mask_dataset.py \
# $(pwd)/configs/data_gen/random_thick_256.yaml \
# datasets/afhq/test/test_origin/cat/ \
# datasets/afhq/test/test_256_with_mask/cat/random_thick_256/

# python3 bin/gen_mask_dataset.py \
# $(pwd)/configs/data_gen/random_thin_256.yaml \
# datasets/afhq/test/test_origin/cat/ \
# datasets/afhq/test/test_256_with_mask/cat/random_thin_256/

# python3 bin/gen_mask_dataset.py \
# $(pwd)/configs/data_gen/random_medium_256.yaml \
# datasets/afhq/test/test_origin/cat/ \
# datasets/afhq/test/test_256_with_mask/cat/random_medium_256/


# python3 bin/gen_mask_dataset.py \
# $(pwd)/configs/data_gen/random_thick_256.yaml \
# /home/mona/codes/lama/datasets/val256crop_100/origin/ \
# /home/mona/codes/lama/datasets/val256crop_100/withMask/random_thick_256/

# -------------------------------------------------------------------------------
# python3 bin/gen_mask_dataset.py \
# $(pwd)/configs/data_gen/random_thick_256.yaml \
# datasets/afhq/train/train256_cat/train256_cat_2000/val256_cat_500_source/ \
# datasets/afhq/train/train256_cat/train256_cat_2000/val/random_thick_256/

# python3 bin/gen_mask_dataset.py \
# $(pwd)/configs/data_gen/random_thin_256.yaml \
# datasets/afhq/train/train256_cat/train256_cat_2000/val256_cat_500_source/ \
# datasets/afhq/train/train256_cat/train256_cat_2000/val/random_thin_256/

# python3 bin/gen_mask_dataset.py \
# $(pwd)/configs/data_gen/random_medium_256.yaml \
# datasets/afhq/train/train256_cat/train256_cat_2000/val256_cat_500_source/ \
# datasets/afhq/train/train256_cat/train256_cat_2000/val/random_medium_256/

# ----------------------------------------------------------------------------------

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thick_256.yaml \
datasets/afhq/train/train256_wild/train256_wild_100_random_seed2/val256_cat_500_source/ \
datasets/afhq/train/train256_wild/train256_wild_100_random_seed2/val/random_thick_256/ \
2

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thin_256.yaml \
datasets/afhq/train/train256_wild/train256_wild_100_random_seed2/val256_cat_500_source/ \
datasets/afhq/train/train256_wild/train256_wild_100_random_seed2/val/random_thin_256/ \
2

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_medium_256.yaml \
datasets/afhq/train/train256_wild/train256_wild_100_random_seed2/val256_cat_500_source/ \
datasets/afhq/train/train256_wild/train256_wild_100_random_seed2/val/random_medium_256/ \
2


python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thick_256.yaml \
datasets/afhq/train/train256_wild/train256_wild_100_random_seed3/val256_cat_500_source/ \
datasets/afhq/train/train256_wild/train256_wild_100_random_seed3/val/random_thick_256/ \
3

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thin_256.yaml \
datasets/afhq/train/train256_wild/train256_wild_100_random_seed3/val256_cat_500_source/ \
datasets/afhq/train/train256_wild/train256_wild_100_random_seed3/val/random_thin_256/ \
3

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_medium_256.yaml \
datasets/afhq/train/train256_wild/train256_wild_100_random_seed3/val256_cat_500_source/ \
datasets/afhq/train/train256_wild/train256_wild_100_random_seed3/val/random_medium_256/ \
3

# python3 bin/gen_mask_dataset.py \
# $(pwd)/configs/data_gen/train_mask_large.yaml \
# datasets/afhq/train_origin/cat/ \
# datasets/afhq/test/training_mask_test/cat-all-large-mask/