# Environment setup

Clone the repo:
`git clone https://github.com/saic-mdal/lama.git`

There are three options of an environment:

1. Python virtualenv:

    ```
    virtualenv inpenv --python=/usr/bin/python3
    source inpenv/bin/activate
    pip install torch==1.8.0 torchvision==0.9.0
    
    cd T2inpainting
    pip install -r requirements.txt 
    ```

2. Conda
    
    ```
    % Install conda for Linux, for other OS download miniconda at https://docs.conda.io/en/latest/miniconda.html
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    $HOME/miniconda/bin/conda init bash

    cd T2inpainting
    conda env create -f conda_env.yml
    conda activate lama
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
    pip install pytorch-lightning==1.2.9
    ```
 



# Train

Make sure you run:

```
cd T2inpainting
export TORCH_HOME=$(pwd) && export PYTHONPATH=.
```

Then download models for _perceptual loss_:

    mkdir -p ade20k/ade20k-resnet50dilated-ppm_deepsup/
    wget -P ade20k/ade20k-resnet50dilated-ppm_deepsup/ http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth


## Create your data

Please check bash scripts for data preparation and mask generation from CelebaHQ section,
if you stuck at one of the following steps.


On the host machine:

    # Make shure you are in lama folder
    cd T2inpainting
    export TORCH_HOME=$(pwd) && export PYTHONPATH=.

    # You need to prepare following image folders:
    $ ls my_dataset
    train
    val_source # 2000 or more images
    visual_test_source # 100 or more images
    eval_source # 2000 or more images

    # LaMa generates random masks for the train data on the flight,
    # but needs fixed masks for test and visual_test for consistency of evaluation.

    # Suppose, we want to evaluate and pick best models 
    # on 512x512 val dataset  with thick/thin/medium masks 
    # And your images have .jpg extention:

    python3 bin/gen_mask_dataset.py \
    $(pwd)/configs/data_gen/random_<size>_512.yaml \ # thick, thin, medium
    my_dataset/val_source/ \
    my_dataset/val/random_<size>_512.yaml \# thick, thin, medium
    --ext jpg

    # So the mask generator will: 
    # 1. resize and crop val images and save them as .png
    # 2. generate masks
    
    ls my_dataset/val/random_medium_512/
    image1_crop000_mask000.png
    image1_crop000.png
    image2_crop000_mask000.png
    image2_crop000.png
    ...

    # Generate thick, thin, medium masks for visual_test folder:

    python3 bin/gen_mask_dataset.py \
    $(pwd)/configs/data_gen/random_<size>_512.yaml \  #thick, thin, medium
    my_dataset/visual_test_source/ \
    my_dataset/visual_test/random_<size>_512/ \ #thick, thin, medium
    --ext jpg
    

    ls my_dataset/visual_test/random_thick_512/
    image1_crop000_mask000.png
    image1_crop000.png
    image2_crop000_mask000.png
    image2_crop000.png
    ...

    # Same process for eval_source image folder:
    
    python3 bin/gen_mask_dataset.py \
    $(pwd)/configs/data_gen/random_<size>_512.yaml \  #thick, thin, medium
    my_dataset/eval_source/ \
    my_dataset/eval/random_<size>_512/ \ #thick, thin, medium
    --ext jpg
    


    # Generate location config file which locate these folders:
    
    touch my_dataset.yaml
    echo "data_root_dir: $(pwd)/my_dataset/" >> my_dataset.yaml
    echo "out_root_dir: $(pwd)/experiments/" my_dataset.yaml
    echo "tb_dir: $(pwd)/tb_logs/" my_dataset.yaml
    mv my_dataset.yaml ${PWD}/configs/training/location/


    # Check data config for consistency with my_dataset folder structure:
    $ cat ${PWD}/configs/training/data/abl-04-256-mh-dist
    ...
    train:
      indir: ${location.data_root_dir}/train
      ...
    val:
      indir: ${location.data_root_dir}/val
      img_suffix: .png
    visual_test:
      indir: ${location.data_root_dir}/visual_test
      img_suffix: .png


    # Run training
    python3 bin/train.py -cn T2inpainting_first_stage location=my_dataset data.batch_size=10

    # Evaluation: LaMa training procedure picks best few models according to 
    # scores on my_dataset/val/ 

    # To evaluate one of your best models (i.e. at epoch=32) 
    # on previously unseen my_dataset/eval do the following 
    # for thin, thick and medium:

    # infer:
    python3 bin/predict.py \
    model.path=$(pwd)/experiments/<user>_<date:time>_lama-fourier_/ \
    indir=$(pwd)/my_dataset/eval/random_<size>_512/ \
    outdir=$(pwd)/inference/my_dataset/random_<size>_512 \
    model.checkpoint=epoch32.ckpt

    # metrics calculation:
    python3 bin/evaluate_predicts.py \
    $(pwd)/configs/eval2_gpu.yaml \
    $(pwd)/my_dataset/eval/random_<size>_512/ \
    $(pwd)/inference/my_dataset/random_<size>_512 \
    $(pwd)/inference/my_dataset/random_<size>_512_metrics.csv

  
## License and Acknowledgement
The code and models in this repo are for research purposes only. Our code is bulit upon [lama](https://github.com/advimman/lama).


<div style="text-align:center" align="center">
<br>
<br>
  <img loading="lazy"  height="50px" src="https://raw.githubusercontent.com/saic-mdal/lama-project/main/docs/img/samsung_ai.png" />
</div>
<br>
<p style="font-weight:normal; font-size: 16pt;text-align:center"align="center"  >Copyright © 2021</p>
<br>
