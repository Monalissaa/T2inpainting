python3 outputs/change_model_name.py \
--model_path $1

MODEL_PATH=$2/$3
PROJECT_NAME=$2
if [[ "$4" = "ukiyoe" ]]
then
    TEST_DATA=ukiyoe/ukiyoe-256/test256_ukiyoe_500_with_mask
elif [[ "$4" = "MetFace" ]]
then
    TEST_DATA=MetFace/test_500_with_mask_random_seed10002
elif [[ "$4" = "church" ]]
then
    TEST_DATA=church/test-2000-source-seed1002-256-with-mask
elif [[ "$4" = "CLEVR" ]]
then
    TEST_DATA=CLEVR/test256-2000-source-seed1002-with-mask
else
    TEST_DATA=afhq/test/test_256_with_mask/$4
fi


# ------------------------- model0 --------------------------
# model0, random_thin
python3 bin/predict.py \
model.path=$(pwd)/experiments/${MODEL_PATH} \
indir=$(pwd)/datasets/${TEST_DATA}/random_thin_256/ \
outdir=$(pwd)/outputs/${PROJECT_NAME}/model0_random_thin_256 model.checkpoint=model0.ckpt

python3 bin/evaluate_predicts.py \
$(pwd)/configs/eval2_gpu.yaml \
$(pwd)/datasets/${TEST_DATA}/random_thin_256/ \
$(pwd)/outputs/${PROJECT_NAME}/model0_random_thin_256 \
$(pwd)/outputs/${PROJECT_NAME}/model0_random_thin_256_metrics.csv

# model0, random_medium
python3 bin/predict.py \
model.path=$(pwd)/experiments/${MODEL_PATH} \
indir=$(pwd)/datasets/${TEST_DATA}/random_medium_256/ \
outdir=$(pwd)/outputs/${PROJECT_NAME}/model0_random_medium_256 model.checkpoint=model0.ckpt

python3 bin/evaluate_predicts.py \
$(pwd)/configs/eval2_gpu.yaml \
$(pwd)/datasets/${TEST_DATA}/random_medium_256/ \
$(pwd)/outputs/${PROJECT_NAME}/model0_random_medium_256 \
$(pwd)/outputs/${PROJECT_NAME}/model0_random_medium_256_metrics.csv

# model0, random_thick
python3 bin/predict.py \
model.path=$(pwd)/experiments/${MODEL_PATH} \
indir=$(pwd)/datasets/${TEST_DATA}/random_thick_256/ \
outdir=$(pwd)/outputs/${PROJECT_NAME}/model0_random_thick_256 model.checkpoint=model0.ckpt

python3 bin/evaluate_predicts.py \
$(pwd)/configs/eval2_gpu.yaml \
$(pwd)/datasets/${TEST_DATA}/random_thick_256/ \
$(pwd)/outputs/${PROJECT_NAME}/model0_random_thick_256 \
$(pwd)/outputs/${PROJECT_NAME}/model0_random_thick_256_metrics.csv


# ------------------------- model1 --------------------------
# model1, random_thin
python3 bin/predict.py \
model.path=$(pwd)/experiments/${MODEL_PATH} \
indir=$(pwd)/datasets/${TEST_DATA}/random_thin_256/ \
outdir=$(pwd)/outputs/${PROJECT_NAME}/model1_random_thin_256 model.checkpoint=model1.ckpt

python3 bin/evaluate_predicts.py \
$(pwd)/configs/eval2_gpu.yaml \
$(pwd)/datasets/${TEST_DATA}/random_thin_256/ \
$(pwd)/outputs/${PROJECT_NAME}/model1_random_thin_256 \
$(pwd)/outputs/${PROJECT_NAME}/model1_random_thin_256_metrics.csv


# model1, random_medium
python3 bin/predict.py \
model.path=$(pwd)/experiments/${MODEL_PATH} \
indir=$(pwd)/datasets/${TEST_DATA}/random_medium_256/ \
outdir=$(pwd)/outputs/${PROJECT_NAME}/model1_random_medium_256 model.checkpoint=model1.ckpt

python3 bin/evaluate_predicts.py \
$(pwd)/configs/eval2_gpu.yaml \
$(pwd)/datasets/${TEST_DATA}/random_medium_256/ \
$(pwd)/outputs/${PROJECT_NAME}/model1_random_medium_256 \
$(pwd)/outputs/${PROJECT_NAME}/model1_random_medium_256_metrics.csv

# model1, random_thick
python3 bin/predict.py \
model.path=$(pwd)/experiments/${MODEL_PATH} \
indir=$(pwd)/datasets/${TEST_DATA}/random_thick_256/ \
outdir=$(pwd)/outputs/${PROJECT_NAME}/model1_random_thick_256 model.checkpoint=model1.ckpt

python3 bin/evaluate_predicts.py \
$(pwd)/configs/eval2_gpu.yaml \
$(pwd)/datasets/${TEST_DATA}/random_thick_256/ \
$(pwd)/outputs/${PROJECT_NAME}/model1_random_thick_256 \
$(pwd)/outputs/${PROJECT_NAME}/model1_random_thick_256_metrics.csv


# ------------------------- last --------------------------
# last, random_thin
python3 bin/predict.py \
model.path=$(pwd)/experiments/${MODEL_PATH} \
indir=$(pwd)/datasets/${TEST_DATA}/random_thin_256/ \
outdir=$(pwd)/outputs/${PROJECT_NAME}/last_random_thin_256 model.checkpoint=last.ckpt

python3 bin/evaluate_predicts.py \
$(pwd)/configs/eval2_gpu.yaml \
$(pwd)/datasets/${TEST_DATA}/random_thin_256/ \
$(pwd)/outputs/${PROJECT_NAME}/last_random_thin_256 \
$(pwd)/outputs/${PROJECT_NAME}/last_random_thin_256_metrics.csv


# last, random_medium
python3 bin/predict.py \
model.path=$(pwd)/experiments/${MODEL_PATH} \
indir=$(pwd)/datasets/${TEST_DATA}/random_medium_256/ \
outdir=$(pwd)/outputs/${PROJECT_NAME}/last_random_medium_256 model.checkpoint=last.ckpt

python3 bin/evaluate_predicts.py \
$(pwd)/configs/eval2_gpu.yaml \
$(pwd)/datasets/${TEST_DATA}/random_medium_256/ \
$(pwd)/outputs/${PROJECT_NAME}/last_random_medium_256 \
$(pwd)/outputs/${PROJECT_NAME}/last_random_medium_256_metrics.csv

# last, random_thick
python3 bin/predict.py \
model.path=$(pwd)/experiments/${MODEL_PATH} \
indir=$(pwd)/datasets/${TEST_DATA}/random_thick_256/ \
outdir=$(pwd)/outputs/${PROJECT_NAME}/last_random_thick_256 model.checkpoint=last.ckpt

python3 bin/evaluate_predicts.py \
$(pwd)/configs/eval2_gpu.yaml \
$(pwd)/datasets/${TEST_DATA}/random_thick_256/ \
$(pwd)/outputs/${PROJECT_NAME}/last_random_thick_256 \
$(pwd)/outputs/${PROJECT_NAME}/last_random_thick_256_metrics.csv

# python3 saicinpainting/evaluation/cal_fid_pids_uids.py \
# --output_name $2 \
# --kind $4