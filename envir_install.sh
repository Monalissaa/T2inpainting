# mkdir ZYB
# cd ZYB/
# git clone https://github.com/Monalissaa/lama-under-limited-data.git

conda create -n lama python=3.6 -y
conda activate lama
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install focal-frequency-loss
pip install BeautifulSoup4
pip install pyyaml
pip install tqdm
pip install numpy
pip install easydict==1.9.0
pip install scikit-image==0.17.2
pip install scikit-learn==0.24.2
pip install opencv-python
pip install tensorflow
pip install joblib
pip install matplotlib
pip install pandas
pip install albumentations==0.5.2
pip install hydra-core==1.1.0
pip install pytorch-lightning==1.2.9
pip install tabulate
pip install kornia==0.5.0 webdataset packaging scikit-learn==0.24.2 wldhx.yadisk-direct

