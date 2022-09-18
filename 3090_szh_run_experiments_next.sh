experiment_name=lama-celebahq_full_config_dog_random_seed3_aug_fix_m_cl2l_cg2l_fix_UD_wo_fm_loss
kind=dog
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
# send email to notice
echo "go go go" | mail -s "3090_szh_lama-transfer-dog-random-seed3-our1" 937315849@qq.com
