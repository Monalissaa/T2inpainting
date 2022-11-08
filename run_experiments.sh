experiment_name=big-lama-place2_full_config_cat_seed1_wo_fm_release_tsa_global_bn
kind=cat
# create experiments dir
mkdir experiments/$experiment_name

# run experiment
python3 bin/train.py -cn $experiment_name

# create outputs dir
mkdir outputs/$experiment_name

# look for the true experiment dir in outputs
path='outputs'
files=$(ls $path)
first_level_need_contain='2022'
first_level_name=''
for filename in $files
do
   if [[ $filename == *$first_level_need_contain* ]]
   then
     first_level_name=$filename
   else
     break
   fi
done

path_second_level=outputs/$first_level_name
files=$(ls $path_second_level)
second_level_name=''
for filename in $files
do
   second_level_name=$filename
done

experiment_dir_in_outputs=outputs/$first_level_name/$second_level_name



# mv experiment in outputs to experiments dir
mv $experiment_dir_in_outputs experiments/$experiment_name/

# run predict & metric
bash outputs/predict\&metric_template.sh $PWD/experiments/$experiment_name/$second_level_name/models $experiment_name $second_level_name $kind
#

output_name=$experiment_name

echo "---------------------thin_mask_results--------------------" >> outputs/$output_name/three_results.txt
cat outputs/$output_name/last_random_thin_256_metrics.csv >> outputs/$output_name/three_results.txt
cat outputs/$output_name/last_thin_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt


cat outputs/$output_name/model0_random_thin_256_metrics.csv >> outputs/$output_name/three_results.txt
cat outputs/$output_name/model0_thin_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt


cat outputs/$output_name/model1_random_thin_256_metrics.csv >> outputs/$output_name/three_results.txt
cat outputs/$output_name/model1_thin_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt


echo "" >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt

echo "---------------------medium_mask_results--------------------" >> outputs/$output_name/three_results.txt
cat outputs/$output_name/last_random_medium_256_metrics.csv >> outputs/$output_name/three_results.txt
cat outputs/$output_name/last_medium_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt


cat outputs/$output_name/model0_random_medium_256_metrics.csv >> outputs/$output_name/three_results.txt
cat outputs/$output_name/model0_medium_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt


cat outputs/$output_name/model1_random_medium_256_metrics.csv >> outputs/$output_name/three_results.txt
cat outputs/$output_name/model1_medium_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt


echo "" >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt

echo "---------------------thick_mask_results--------------------" >> outputs/$output_name/three_results.txt
cat outputs/$output_name/last_random_thick_256_metrics.csv >> outputs/$output_name/three_results.txt
cat outputs/$output_name/last_thick_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt


cat outputs/$output_name/model0_random_thick_256_metrics.csv >> outputs/$output_name/three_results.txt
cat outputs/$output_name/model0_thick_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt


cat outputs/$output_name/model1_random_thick_256_metrics.csv >> outputs/$output_name/three_results.txt
cat outputs/$output_name/model1_thick_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt

mv experiments/$experiment_name /mnt/d/post/codes/lama/experiment/

# send email to notice
echo "go go go 714_cat_seed1_big-lama" | mail -s "714_cat_seed1_big-lama!" 937315849@qq.com


# # # ########################### second


experiment_name=big-lama-place2_full_config_church_seed1_wo_fm_release_tsa_global_bn
kind=church
# create experiments dir
mkdir experiments/$experiment_name

# run experiment
python3 bin/train.py -cn $experiment_name

# create outputs dir
mkdir outputs/$experiment_name

# look for the true experiment dir in outputs
path='outputs'
files=$(ls $path)
first_level_need_contain='2022'
first_level_name=''
for filename in $files
do
   if [[ $filename == *$first_level_need_contain* ]]
   then
     first_level_name=$filename
   else
     break
   fi
done

path_second_level=outputs/$first_level_name
files=$(ls $path_second_level)
second_level_name=''
for filename in $files
do
   second_level_name=$filename
done

experiment_dir_in_outputs=outputs/$first_level_name/$second_level_name



# mv experiment in outputs to experiments dir
mv $experiment_dir_in_outputs experiments/$experiment_name/

# run predict & metric
bash outputs/predict\&metric_template.sh $PWD/experiments/$experiment_name/$second_level_name/models $experiment_name $second_level_name $kind
#

output_name=$experiment_name

echo "---------------------thin_mask_results--------------------" >> outputs/$output_name/three_results.txt
cat outputs/$output_name/last_random_thin_256_metrics.csv >> outputs/$output_name/three_results.txt
cat outputs/$output_name/last_thin_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt


cat outputs/$output_name/model0_random_thin_256_metrics.csv >> outputs/$output_name/three_results.txt
cat outputs/$output_name/model0_thin_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt


cat outputs/$output_name/model1_random_thin_256_metrics.csv >> outputs/$output_name/three_results.txt
cat outputs/$output_name/model1_thin_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt


echo "" >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt

echo "---------------------medium_mask_results--------------------" >> outputs/$output_name/three_results.txt
cat outputs/$output_name/last_random_medium_256_metrics.csv >> outputs/$output_name/three_results.txt
cat outputs/$output_name/last_medium_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt


cat outputs/$output_name/model0_random_medium_256_metrics.csv >> outputs/$output_name/three_results.txt
cat outputs/$output_name/model0_medium_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt


cat outputs/$output_name/model1_random_medium_256_metrics.csv >> outputs/$output_name/three_results.txt
cat outputs/$output_name/model1_medium_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt


echo "" >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt

echo "---------------------thick_mask_results--------------------" >> outputs/$output_name/three_results.txt
cat outputs/$output_name/last_random_thick_256_metrics.csv >> outputs/$output_name/three_results.txt
cat outputs/$output_name/last_thick_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt


cat outputs/$output_name/model0_random_thick_256_metrics.csv >> outputs/$output_name/three_results.txt
cat outputs/$output_name/model0_thick_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt


cat outputs/$output_name/model1_random_thick_256_metrics.csv >> outputs/$output_name/three_results.txt
cat outputs/$output_name/model1_thick_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
echo "" >> outputs/$output_name/three_results.txt

mv experiments/$experiment_name /mnt/d/post/codes/lama/experiment/

# send email to notice
echo "go go go 714_church_seed1_big_lama" | mail -s "714_church_seed1_big_lama!" 937315849@qq.com


# ########################### third


# experiment_name=big-lama-place2_full_config_church_seed2_wo_fm_tsa_g2g_best_test_stage_two_lr01_fix_alpha
# kind=church
# # create experiments dir
# mkdir experiments/$experiment_name

# # run experiment
# python3 bin/train.py -cn $experiment_name

# # create outputs dir
# mkdir outputs/$experiment_name

# # look for the true experiment dir in outputs
# path='outputs'
# files=$(ls $path)
# first_level_need_contain='2022'
# first_level_name=''
# for filename in $files
# do
#    if [[ $filename == *$first_level_need_contain* ]]
#    then
#      first_level_name=$filename
#    else
#      break
#    fi
# done

# path_second_level=outputs/$first_level_name
# files=$(ls $path_second_level)
# second_level_name=''
# for filename in $files
# do
#    second_level_name=$filename
# done

# experiment_dir_in_outputs=outputs/$first_level_name/$second_level_name



# # mv experiment in outputs to experiments dir
# mv $experiment_dir_in_outputs experiments/$experiment_name/

# # run predict & metric
# bash outputs/predict\&metric_template.sh $PWD/experiments/$experiment_name/$second_level_name/models $experiment_name $second_level_name $kind
# #

# output_name=$experiment_name

# echo "---------------------thin_mask_results--------------------" >> outputs/$output_name/three_results.txt
# cat outputs/$output_name/last_random_thin_256_metrics.csv >> outputs/$output_name/three_results.txt
# cat outputs/$output_name/last_thin_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
# echo "" >> outputs/$output_name/three_results.txt
# echo "" >> outputs/$output_name/three_results.txt


# cat outputs/$output_name/model0_random_thin_256_metrics.csv >> outputs/$output_name/three_results.txt
# cat outputs/$output_name/model0_thin_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
# echo "" >> outputs/$output_name/three_results.txt
# echo "" >> outputs/$output_name/three_results.txt


# cat outputs/$output_name/model1_random_thin_256_metrics.csv >> outputs/$output_name/three_results.txt
# cat outputs/$output_name/model1_thin_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
# echo "" >> outputs/$output_name/three_results.txt
# echo "" >> outputs/$output_name/three_results.txt


# echo "" >> outputs/$output_name/three_results.txt
# echo "" >> outputs/$output_name/three_results.txt

# echo "---------------------medium_mask_results--------------------" >> outputs/$output_name/three_results.txt
# cat outputs/$output_name/last_random_medium_256_metrics.csv >> outputs/$output_name/three_results.txt
# cat outputs/$output_name/last_medium_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
# echo "" >> outputs/$output_name/three_results.txt
# echo "" >> outputs/$output_name/three_results.txt


# cat outputs/$output_name/model0_random_medium_256_metrics.csv >> outputs/$output_name/three_results.txt
# cat outputs/$output_name/model0_medium_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
# echo "" >> outputs/$output_name/three_results.txt
# echo "" >> outputs/$output_name/three_results.txt


# cat outputs/$output_name/model1_random_medium_256_metrics.csv >> outputs/$output_name/three_results.txt
# cat outputs/$output_name/model1_medium_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
# echo "" >> outputs/$output_name/three_results.txt
# echo "" >> outputs/$output_name/three_results.txt


# echo "" >> outputs/$output_name/three_results.txt
# echo "" >> outputs/$output_name/three_results.txt

# echo "---------------------thick_mask_results--------------------" >> outputs/$output_name/three_results.txt
# cat outputs/$output_name/last_random_thick_256_metrics.csv >> outputs/$output_name/three_results.txt
# cat outputs/$output_name/last_thick_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
# echo "" >> outputs/$output_name/three_results.txt
# echo "" >> outputs/$output_name/three_results.txt


# cat outputs/$output_name/model0_random_thick_256_metrics.csv >> outputs/$output_name/three_results.txt
# cat outputs/$output_name/model0_thick_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
# echo "" >> outputs/$output_name/three_results.txt
# echo "" >> outputs/$output_name/three_results.txt


# cat outputs/$output_name/model1_random_thick_256_metrics.csv >> outputs/$output_name/three_results.txt
# cat outputs/$output_name/model1_thick_fid_pids_uids.txt >> outputs/$output_name/three_results.txt
# echo "" >> outputs/$output_name/three_results.txt

# mv experiments/$experiment_name /mnt/d/post/codes/lama/experiment/

# # send email to notice
# echo "go go go 714_ch_big_tsa_g2g_seed2_best_test_stage_two_fix_alpha_lr01" | mail -s "714_ch_big_tsa_g2g_seed2_best_test_stage_two_fix_alpha_lr01!" 937315849@qq.com