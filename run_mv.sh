mv experiments/lama-celebahq_full_config_cat_horizontal_flip_transfer_data2000_trainx3 /mnt/d/post/codes/lama/experiment/

# mv experiments/lama-transfer-aug_fix_middleBlocks_convl2l_convg2l_fix_UpDown_without_featureMatching_loss_stage_two_horizontal_flip_no_fix_without_featureMatching_loss_with_l1_weight60_loss /mnt/d/post/codes/lama/experiment/

# mv experiments/lama-transfer-aug_fix_middleBlocks_convl2l_convg2l_fix_UpDown_without_featureMatching_loss_stage_two_horizontal_flip_no_fix_lrx001_without_featureMatching_loss_with_l1_weight60_loss /mnt/d/post/codes/lama/experiment/



# python3 bin/train.py -cn lama-celebahq_full_config_data500

# python3 bin/train.py -cn lama-celebahq_full_config_data1000

# python3 bin/train.py -cn lama-celebahq_full_config_aug_ffl

# python3 bin/train.py -cn lama-celebahq_full_config_aug_tv

# python3 bin/train.py -cn lama-celebahq_full_config_aug_ffl_tv
