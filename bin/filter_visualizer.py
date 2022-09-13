from fastai.conv_learner import *
import cv2
from cv2 import resize
import PIL
import matplotlib.pyplot as plt
import torch
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import set_requires_grad
import numpy as np

import hydra
from omegaconf import OmegaConf

# %matplotlib inline

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output,requires_grad=True).cuda()
    def close(self):
        self.hook.remove()

class FilterVisualizer():
    def __init__(self, size=56, upscaling_steps=12, upscaling_factor=1.2, inpaint_model=None):
        self.size, self.upscaling_steps, self.upscaling_factor = size, upscaling_steps, upscaling_factor
        # self.model = vgg16(pre=True).cuda().eval()
        self.model = inpaint_model.cuda().eval()
        
        set_requires_grad(self.model, False)

    def visualize(self, layer, filter, lr=0.1, opt_steps=20, blur=None):
        sz = self.size
        img = np.uint8(np.random.uniform(150, 180, (sz, sz, 3)))/255  # generate random image
        activations = SaveFeatures(list(self.model.children())[layer])  # register hook

        for _ in range(self.upscaling_steps):  # scale the image up upscaling_steps times
            train_tfms, val_tfms = tfms_from_model(self.model, sz)
            img_var = V(val_tfms(img)[None], requires_grad=True)  # convert image to Variable that requires grad
            optimizer = torch.optim.Adam([img_var], lr=lr, weight_decay=1e-6)
            for n in range(opt_steps):  # optimize pixel values for opt_steps times
                optimizer.zero_grad()
                self.model(img_var)
                loss = -activations.features[0, filter].mean()
                loss.backward()
                optimizer.step()
            img = val_tfms.denorm(img_var.data.cpu().numpy()[0].transpose(1,2,0))
            self.output = img
            sz = int(self.upscaling_factor * sz)  # calculate new image size
            img = cv2.resize(img, (sz, sz), interpolation = cv2.INTER_CUBIC)  # scale image up
            if blur is not None: img = cv2.blur(img,(blur,blur))  # blur image to reduce high frequency patterns
        self.save(layer, filter)
        activations.close()
        
    def save(self, layer, filter):
        plt.imsave("layer_"+str(layer)+"_filter_"+str(filter)+".jpg", np.clip(self.output, 0, 1))


@hydra.main(config_path='../configs/training', config_name='tiny_test.yaml')
def main(config: OmegaConf):


    # training_model = make_training_model(config)
    # checkpoint_path = '/home/mona/codes/lama/experiments/celeb_hq_all_256_lama_from_scratch/mona_2022-01-11_11-09-25_train_lama-fourier-celeba_/models/best.ckpt'
    checkpoint_path = '/home/mona/codes/lama/experiments/big-lama-transfer-reInitUp/10-36-13/models/best.ckpt'
    # checkpoint_path = '/home/mona/codes/lama/experiments/big-lama-with-discr/models/best.ckpt'
    training_model = load_checkpoint(config, checkpoint_path)

    layer = 40
    filter = 265

    FV = FilterVisualizer(size=56, upscaling_steps=12, upscaling_factor=1.2, inpaint_model = training_model)
    FV.visualize(layer, filter, blur=5)

    img = PIL.Image.open("layer_"+str(layer)+"_filter_"+str(filter)+".jpg")
    plt.figure(figsize=(7,7))
    plt.imshow(img)


if __name__ == '__main__':
    main()