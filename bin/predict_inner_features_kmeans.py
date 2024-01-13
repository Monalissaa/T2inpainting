#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import logging
import os
import sys
import traceback

from saicinpainting.evaluation.utils import move_to_device

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint, DefaultInpaintingTrainingModule
from saicinpainting.utils import register_debug_signal_handlers, get_shape

from kmeans_pytorch import kmeans
import torch.nn.functional as F
import torchvision

LOGGER = logging.getLogger(__name__)


# @hydra.main(config_path='../configs/prediction', config_name='default_inner_features.yaml')
@hydra.main(config_path='../configs/prediction', config_name='default_modify_our.yaml')
def main(predict_config: OmegaConf):
    try:
        register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device(predict_config.device)

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        checkpoint_path = os.path.join(predict_config.model.path, 'models', predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False)
        model.freeze()
        model.to(device)

        assert isinstance(model, DefaultInpaintingTrainingModule), 'Only DefaultInpaintingTrainingModule is supported'
        assert isinstance(getattr(model.generator, 'model', None), torch.nn.Sequential)

        if not predict_config.indir.endswith('/'):
            predict_config.indir += '/'

        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)

        max_level = max(predict_config.levels)

        with torch.no_grad():
            for img_i in tqdm.trange(len(dataset)):
                mask_fname = dataset.mask_filenames[img_i]
                cur_out_fname = os.path.join(predict_config.outdir, os.path.splitext(mask_fname[len(predict_config.indir):])[0])
                os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)

                batch = move_to_device(default_collate([dataset[img_i]]), device)

                img = batch['image']
                mask = batch['mask']
                mask[:] = 0
                mask_h, mask_w = mask.shape[-2:]
                mask[:, :,
                    mask_h // 2 - predict_config.hole_radius : mask_h // 2 + predict_config.hole_radius,
                    mask_w // 2 - predict_config.hole_radius : mask_w // 2 + predict_config.hole_radius] = 1

                masked_img = torch.cat([img * (1 - mask), mask], dim=1)

                feats = masked_img
                vis_img = []
                for level_i, level in enumerate(model.generator.model):
                    feats = level(feats)
                    if level_i in predict_config.levels:
                        # if level_i>4 and level_i<14:
                        #     cur_feats = feats[1]
                        # else:
                        #     cur_feats = torch.cat([f for f in feats if torch.is_tensor(f)], dim=1) \
                        #         if isinstance(feats, tuple) else feats
                        # all
                        # cur_feats = torch.cat([f for f in feats if torch.is_tensor(f)], dim=1) \
                        #     if isinstance(feats, tuple) else feats
                        # global
                        # cur_feats = feats[1]
                        # local
                        cur_feats = feats[0]
                        cur_feats = get_cluster_vis(cur_feats, num_clusters=6,
                                              target_res=cur_feats.shape[-1])
                        vis_img.append(cur_feats)
                        # cur_feats = (cur_feats + 1) * 127.5 / 255.0
                        # torchvision.utils.save_image(cur_feats, f'{cur_out_fname}_{level_i}.png')

                        # if predict_config.slice_channels:
                        #     cur_feats = cur_feats[:, slice(*predict_config.slice_channels)]
                        #
                        # cur_feat = cur_feats.pow(2).mean(1).pow(0.5).clone()
                        # cur_feat -= cur_feat.min()
                        # cur_feat /= cur_feat.std()
                        # cur_feat = cur_feat.clamp(0, 1) / 1
                        # cur_feat = cur_feat.cpu().numpy()[0]
                        # cur_feat *= 255
                        # cur_feat = np.clip(cur_feat, 0, 255).astype('uint8')
                        # cv2.imwrite(cur_out_fname + f'_lev{level_i:02d}_norm_1.png', cur_feat)

                        # for channel_i in predict_config.channels:
                        #
                        #     cur_feat = cur_feats[0, channel_i].clone().detach().cpu().numpy()
                        #     cur_feat -= cur_feat.min()
                        #     cur_feat /= cur_feat.max()
                        #     cur_feat *= 255
                        #     cur_feat = np.clip(cur_feat, 0, 255).astype('uint8')
                        #     cv2.imwrite(cur_out_fname + f'_lev{level_i}_ch{channel_i}.png', cur_feat)
                    elif level_i >= max_level:
                        break

                for idx, val in enumerate(vis_img):
                    vis_img[idx] = F.interpolate(val, size=(256, 256))

                vis_img = torch.cat(vis_img, dim=0)  # bnum * res_num, 256, 256
                vis_img = (vis_img + 1) * 127.5 / 255.0
                vis_img = torchvision.utils.make_grid(vis_img, normalize=False, nrow=9)
                # torchvision.utils.save_image(vis_img, f'{cur_out_fname}_combine_local.svg')
                torchvision.utils.save_image(vis_img, f'{cur_out_fname}_combine_local.eps')
                if img_i>10:
                    break
                # for res in target_layers:
                #     img = get_cluster_vis(fake_feat[res], num_clusters=num_clusters,
                #                           target_res=res)  # bnum, 256, 256
                #     vis_img.append(img)
                #
                # for idx, val in enumerate(vis_img):
                #     vis_img[idx] = F.interpolate(val, size=(256, 256))
                #
                # vis_img = torch.cat(vis_img, dim=0)  # bnum * res_num, 256, 256
                # vis_img = (vis_img + 1) * 127.5 / 255.0
                # fake_imgs = (fake_imgs + 1) * 127.5 / 255.0
                # fake_imgs = F.interpolate(fake_imgs, size=(256, 256))
                #
                # vis_img = torch.cat([fake_imgs, vis_img], dim=0)
                # vis_img = torchvision.utils.make_grid(vis_img, normalize=False, nrow=batch_size)
                # torchvision.utils.save_image(vis_img, f'{outdir}/{iter_idx}.png')
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


def get_colors():
    dummy_color = np.array([
        [178, 34, 34],  # firebrick
        [0, 139, 139],  # dark cyan
        [245, 222, 179],  # wheat
        [25, 25, 112],  # midnight blue
        [255, 140, 0],  # dark orange
        [128, 128, 0],  # olive
        [50, 50, 50],  # dark grey
        [34, 139, 34],  # forest green
        [100, 149, 237],  # corn flower blue
        [153, 50, 204],  # dark orchid
        [240, 128, 128],  # light coral
    ])

    for t in (0.6, 0.3):  # just increase the number of colors for big K
        dummy_color = np.concatenate((dummy_color, dummy_color * t))

    dummy_color = (np.array(dummy_color) - 128.0) / 128.0
    dummy_color = torch.from_numpy(dummy_color)

    return dummy_color


def get_cluster_vis(feat, num_clusters=10, target_res=16):
    # feat : NCHW
    print(feat.size())
    img_num, C, H, W = feat.size()
    feat = feat.permute(0, 2, 3, 1).contiguous().view(img_num * H * W, -1)
    feat = feat.to(torch.float32).cuda()
    cluster_ids_x, cluster_centers = kmeans(
        X=feat, num_clusters=num_clusters, distance='cosine',
        tol=1e-4,
        device=torch.device("cuda:0"))

    cluster_ids_x = cluster_ids_x.cuda()
    cluster_centers = cluster_centers.cuda()
    color_rgb = get_colors().cuda()
    vis_img = []
    for idx in range(img_num):
        num_pixel = target_res * target_res
        current_res = cluster_ids_x[num_pixel * idx:num_pixel * (idx + 1)].cuda()
        color_ids = torch.index_select(color_rgb, 0, current_res)
        color_map = color_ids.permute(1, 0).view(1, 3, target_res, target_res)
        color_map = F.interpolate(color_map, size=(256, 256))
        vis_img.append(color_map.cuda())

    vis_img = torch.cat(vis_img, dim=0)

    return vis_img



if __name__ == '__main__':
    main()
