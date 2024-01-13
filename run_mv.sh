# mv experiments/lama-celebahq_full_config_church_data100_seed3_aug_fix_m_cl2l_cg2l_fix_UD_wo_fm_loss_stage_two_aug_tsa_all /mnt/d/post/codes/lama/experiment/

# mv experiments/lama-celebahq_full_config_cat_aug_only_release_ten_blocks /mnt/d/post/codes/lama/experiment/


# mv experiments/lama-transfer-aug_fix_middleBlocks_convl2l_convg2l_fix_UpDown_without_featureMatching_loss_stage_two_horizontal_flip_no_fix_without_featureMatching_loss_with_l1_weight60_loss /mnt/d/post/codes/lama/experiment/

# mv experiments/lama-transfer-aug_fix_middleBlocks_convl2l_convg2l_fix_UpDown_without_featureMatching_loss_stage_two_horizontal_flip_no_fix_lrx001_without_featureMatching_loss_with_l1_weight60_loss /mnt/d/post/codes/lama/experiment/



# python3 bin/train.py -cn lama-celebahq_full_config_data500

# python3 bin/train.py -cn lama-celebahq_full_config_data1000

# python3 bin/train.py -cn lama-celebahq_full_config_aug_ffl

# python3 bin/train.py -cn lama-celebahq_full_config_aug_tv

# python3 bin/train.py -cn lama-celebahq_full_config_aug_ffl_tv

mv experiments/lama-celebahq_full_config_dog_random_seed0_aug_fix_m_cl2l_cg2l_fix_UD_wo_fm_loss_stage_two_aug_tsa_all_tsa_alpha5_20240111 /mnt/d/post/codes/lama/experiment/

mv experiments/lama-celebahq_full_config_dog_random_seed0_aug_fix_m_cl2l_cg2l_fix_UD_wo_fm_loss_stage_two_aug_tsa_all_tsa_alpha05_20240111_ckpt_destroy /mnt/d/post/codes/lama/experiment/

# mv outputs/big-lama-place2_full_config_cat_seed1 /mnt/d/post/codes/lama/outputs/

# mv outputs/big-lama-place2_full_config_cat_seed1_wo_fm_tsa_all /mnt/d/post/codes/lama/outputs/

# mv outputs/big-lama-place2_full_config_cat_seed2_aug_fix_cl2l_cg2l_UD_wo_fm_stage_two_aug_tsa_all /mnt/d/post/codes/lama/outputs/

# mv outputs/big-lama-place2_full_config_cat_seed2_wo_fm_tsa_g2g /mnt/d/post/codes/lama/outputs/

# mv outputs/big-lama-place2_full_config_cat_seed2_wo_fm_tsa_g2g_stage_two_fix_alpha /mnt/d/post/codes/lama/outputs/

# mv outputs/big-lama-place2_full_config_cat_seed3_aug_wo_fm_tsa_all_1_1 /mnt/d/post/codes/lama/outputs/

# mv outputs/big-lama-place2_full_config_cat_seed3_aug_wo_fm_tsa_one_conv_true /mnt/d/post/codes/lama/outputs/

# mv outputs/lama-celebahq_full_config_MetFace_random_seed1_horizontal_flip_fix_m_cl2l_cg2l_UD_wo_fm_weight_decay_lambda01 /mnt/d/post/codes/lama/outputs/



# mv outputs/lama-celebahq_full_config_cat_100_transfer_fsmr4blocks /mnt/d/post/codes/lama/outputs/

# mv outputs/lama-celebahq_full_config_cat_100_transfer_fsmr4blocks_aug /mnt/d/post/codes/lama/outputs/

dir=outputs/
for folder in $dir*/; do
  # do something with folder
  mv $folder /mnt/d/post/codes/lama/outputs/
done
