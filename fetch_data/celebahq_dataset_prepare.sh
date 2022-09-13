mkdir celeba-hq-dataset

unzip data256x256.zip -d celeba-hq-dataset/

# Reindex
for i in `echo {00001..30000}`
do
#    mv 'celeba-hq-dataset/data256x256/'$i'.jpg' 'celeba-hq-dataset/data256x256/'$[10#$i - 1]'.jpg'
    mv 'datasets/celeba-hq-dataset/data256x256/'$i'.jpg' 'datasets/celeba-hq-dataset/data256x256/'$[10#$i - 1]'.jpg'
done


# Split: split train -> train & val
cat fetch_data/train_shuffled.flist | shuf > datasets/celeba-hq-dataset/temp_train_shuffled.flist
cat datasets/celeba-hq-dataset/temp_train_shuffled.flist | head -n 2000 > datasets/celeba-hq-dataset/val_shuffled.flist
cat datasets/celeba-hq-dataset/temp_train_shuffled.flist | tail -n +2001 > datasets/celeba-hq-dataset/train_shuffled.flist
cat fetch_data/val_shuffled.flist > datasets/celeba-hq-dataset/visual_test_shuffled.flist

mkdir datasets/celeba-hq-dataset/train_256/
mkdir datasets/celeba-hq-dataset/val_source_256/
mkdir datasets/celeba-hq-dataset/visual_test_source_256/

cat datasets/celeba-hq-dataset/train_shuffled.flist | xargs -I {} mv datasets/celeba-hq-dataset/data256x256/{} datasets/celeba-hq-dataset/train_256/
cat datasets/celeba-hq-dataset/val_shuffled.flist | xargs -I {} mv datasets/celeba-hq-dataset/data256x256/{} datasets/celeba-hq-dataset/val_source_256/
cat datasets/celeba-hq-dataset/visual_test_shuffled.flist | xargs -I {} mv datasets/celeba-hq-dataset/data256x256/{} datasets/celeba-hq-dataset/visual_test_source_256/


# create location config celeba.yaml
PWD=$(pwd)
DATASET=${PWD}/datasets/celeba-hq-dataset
CELEBA=${PWD}/configs/training/location/celeba.yaml

touch $CELEBA
echo "# @package _group_" >> $CELEBA
echo "data_root_dir: ${DATASET}/" >> $CELEBA
echo "out_root_dir: ${PWD}/experiments/" >> $CELEBA
echo "tb_dir: ${PWD}/tb_logs/" >> $CELEBA
echo "pretrained_models: ${PWD}/" >> $CELEBA
