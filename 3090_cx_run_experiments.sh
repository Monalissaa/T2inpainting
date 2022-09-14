# python3 bin/train.py -cn lama-celebahq_full_config_dog_aug_fix_middleBlocks_convl2l_convg2l_fix_UpDown_without_featureMatching_loss

# python3 bin/train.py -cn lama-celebahq_full_config_dog_aug_fix_middleBlocks_convl2l_convg2l_fix_UpDown_without_featureMatching_loss_stage_two_aug_no_fix_without_featureMatching_loss

# python3 bin/train.py -cn lama-celebahq_full_config_dog_aug

# python3 bin/train.py -cn lama-celebahq_full_config_wild_aug_fix_middleBlocks_convl2l_convg2l_fix_UpDown_without_featureMatching_loss_stage_two_aug_no_fix_without_featureMatching_loss

# bash run_predict.sh

# python3 bin/train.py -cn lama-celebahq_full_config_ukiyoe_aug_fix_middleBlocks_convl2l_convg2l_fix_UpDown_without_featureMatching_loss

# bash run_predict.sh

python3 bin/train.py -cn lama-celebahq_full_config_wild_random_seed2.yaml

bash 3090_cx_run_predict.sh

echo "go go go" | mail -s "3090_cx-run-lama_transfer_wild_100_random_seed2-completed!" 937315849@qq.com


