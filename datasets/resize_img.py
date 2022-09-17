import os
# import argparse
import shutil
from PIL import Image
from tqdm import tqdm

source_path = '/home/mona/codes/lama/datasets/MetFace/images'
target_path = '/home/mona/codes/lama/datasets/MetFace/all-256'

target_size = (256, 256)

files = os.listdir(source_path)
for file in tqdm(files):
    image = Image.open(os.path.join(source_path, file))
    image=image.resize(target_size)
    image.save(os.path.join(target_path, file))
