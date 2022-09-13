import numpy as np
import random
from PIL import Image
import os
from tqdm import tqdm

image_source_dir = "/mnt/d/Downloads/val_large/val_large"
image_target_dir = "/mnt/d/Downloads/val_large/val256crop"

def random_crop(image, crop_shape, padding=None):
    oshape = image.size

    # if padding:
    oshape_pad = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    img_pad = Image.new("RGB", (oshape_pad[0], oshape_pad[1]))
    img_pad.paste(image, (padding, padding))
    
    nh = random.randint(0, oshape_pad[0] - crop_shape[0])
    nw = random.randint(0, oshape_pad[1] - crop_shape[1])
    image_crop = img_pad.crop((nh, nw, nh+crop_shape[0], nw+crop_shape[1]))

    return image_crop
    # else:
    #     print("WARNING!!! nothing to do!!!")
    #     return image

    
if __name__ == "__main__":
    image_list = os.listdir(image_source_dir)
    for image in tqdm(image_list[29957+1915:]):

        image_src = Image.open(os.path.join(image_source_dir, image))
        crop_width = 256
        crop_height = 256
        image_dst_crop = random_crop(image_src, [crop_width, crop_height], padding=0)
        image_dst_crop.save(os.path.join(image_target_dir, image))
    
