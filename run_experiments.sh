experiment_name=lama-celebahq_full_config_cat_horizontal_flip_transfer_data2000_trainx3_fix_cl2l_cg2l_UD_wo_fm_stage_two_no_fix_wo_fm
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
echo "go go go 3090_714_cat2000_horizontal_flip_our2_trainx3" | mail -s "3090_714_cat2000_horizontal_flip_our2_trainx3!" 937315849@qq.com

# # ########################### second


experiment_name=lama-celebahq_full_config_cat_horizontal_flip_transfer_data2000_trainx3_from_scratch
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
echo "go go go 3090_714_cat2000_horizontal_flip_from_scratch_trainx3" | mail -s "3090_714_cat2000_horizontal_flip_from_scratch_trainx3!" 937315849@qq.com