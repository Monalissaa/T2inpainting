import logging

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from saicinpainting.training.data.datasets import make_constant_area_crop_params
from saicinpainting.training.losses.distance_weighting import make_mask_distance_weighter
from saicinpainting.training.losses.feature_matching import feature_matching_loss, masked_l1_loss
from saicinpainting.training.modules.fake_fakes import FakeFakesGenerator
from saicinpainting.training.trainers.base import BaseInpaintingTrainingModule, make_multiscale_noise
from saicinpainting.utils import add_prefix_to_keys, get_ramp

from saicinpainting.utils import gumbel_softmax

from saicinpainting.training.data.datasets import make_default_train_dataloader
from saicinpainting.utils import set_requires_grad

from torch import autograd
from torch.autograd import Variable
import re
from numpy import mean
from matplotlib import pyplot as plt 
# from torchvision import transforms
# import time

LOGGER = logging.getLogger(__name__)


def make_constant_area_crop_batch(batch, **kwargs):
    crop_y, crop_x, crop_height, crop_width = make_constant_area_crop_params(img_height=batch['image'].shape[2],
                                                                             img_width=batch['image'].shape[3],
                                                                             **kwargs)
    batch['image'] = batch['image'][:, :, crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
    batch['mask'] = batch['mask'][:, :, crop_y: crop_y + crop_height, crop_x: crop_x + crop_width]
    return batch


class DefaultInpaintingTrainingModule(BaseInpaintingTrainingModule):
    def __init__(self, *args, concat_mask=True, rescale_scheduler_kwargs=None, image_to_discriminator='predicted_image',
                 add_noise_kwargs=None, noise_fill_hole=False, const_area_crop_kwargs=None,
                 distance_weighter_kwargs=None, distance_weighted_mask_for_discr=False,
                 fake_fakes_proba=0, fake_fakes_generator_kwargs=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.concat_mask = concat_mask
        self.rescale_size_getter = get_ramp(**rescale_scheduler_kwargs) if rescale_scheduler_kwargs is not None else None
        self.image_to_discriminator = image_to_discriminator
        self.add_noise_kwargs = add_noise_kwargs
        self.noise_fill_hole = noise_fill_hole
        self.const_area_crop_kwargs = const_area_crop_kwargs
        self.refine_mask_for_losses = make_mask_distance_weighter(**distance_weighter_kwargs) \
            if distance_weighter_kwargs is not None else None
        self.distance_weighted_mask_for_discr = distance_weighted_mask_for_discr
        # print(self.distance_weighted_mask_for_discr)
        # exit(0)

        self.fake_fakes_proba = fake_fakes_proba
        if self.fake_fakes_proba > 1e-3:
            self.fake_fakes_gen = FakeFakesGenerator(**(fake_fakes_generator_kwargs or {}))
        
        # self.estimate_fisher()


    def forward(self, batch, mode='test'):

        if self.training and self.rescale_size_getter is not None:
            cur_size = self.rescale_size_getter(self.global_step)
            batch['image'] = F.interpolate(batch['image'], size=cur_size, mode='bilinear', align_corners=False)
            batch['mask'] = F.interpolate(batch['mask'], size=cur_size, mode='nearest')
            # zhe li bu hui da dao
            # no_mask = torch.zeros(batch['mask'].shape, device='cuda:0') # for first stage with no mask
        
        if self.training and self.const_area_crop_kwargs is not None:
            batch = make_constant_area_crop_batch(batch, **self.const_area_crop_kwargs)

        if self.training and self.config.new_params.contrast>0:
            batch['image'] = torch.cat([batch['image'], batch['image']], dim=0)
            batch['mask'] = torch.cat([batch['mask'], batch['mask_key']], dim=0)
            

        img = batch['image']
        mask = batch['mask']
        if self.config.new_params.two_stage:
            no_mask = torch.zeros(batch['mask'].shape, device='cuda:0') # for first stage with no mask


        masked_img = img * (1 - mask)

        if self.add_noise_kwargs is not None:
            noise = make_multiscale_noise(masked_img, **self.add_noise_kwargs)
            if self.noise_fill_hole:
                masked_img = masked_img + mask * noise[:, :masked_img.shape[1]]
            masked_img = torch.cat([masked_img, noise], dim=1)

        if self.concat_mask:
            masked_img = torch.cat([masked_img, mask], dim=1)
            if self.config.new_params.two_stage:
                no_masked_img = torch.cat([img, no_mask], dim=1)

        if self.config.new_params.two_stage:
            batch['predicted_image'] = self.generator(masked_img, no_masked_img)
            if self.generator.alpha==0: # for first stage with no mask
                batch['inpainted'] = batch['predicted_image']
            else:
                batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']
        elif self.config.new_params.spottune:
            
            probs = self.agent(masked_img)
            
            action = gumbel_softmax(probs.view(probs.size(0), -1, 2))
            policy = action[:,:,1]

            batch['predicted_image'] = self.generator(masked_img, policy)
            batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']
        elif self.config.new_params.CWD>0:
            if self.training:
                batch['acti_model'], batch['acti_small_model'], batch['predicted_image'] = self.generator(masked_img, self.training)
            else: 
                batch['predicted_image'] = self.generator(masked_img, self.training)
            batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']
        elif self.config.new_params.fsmr.blocks>0 and mode=='train':
            batch['predicted_image'] = self.generator(masked_img, use_fsmr=True, fsmr_blocks=self.config.new_params.fsmr.blocks)
            batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']
        else:
            batch['predicted_image'] = self.generator(masked_img)
            batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']
        

        if self.fake_fakes_proba > 1e-3:
            if self.training and torch.rand(1).item() < self.fake_fakes_proba:
                batch['fake_fakes'], batch['fake_fakes_masks'] = self.fake_fakes_gen(img, mask)
                batch['use_fake_fakes'] = True
            else:
                batch['fake_fakes'] = torch.zeros_like(img)
                batch['fake_fakes_masks'] = torch.zeros_like(mask)
                batch['use_fake_fakes'] = False
    
        batch['mask_for_losses'] = self.refine_mask_for_losses(img, batch['predicted_image'], mask) \
            if self.refine_mask_for_losses is not None and self.training \
            else mask

        return batch

    def generator_loss(self, batch):
        img = batch['image']
        predicted_img = batch[self.image_to_discriminator]
        original_mask = batch['mask']
        supervised_mask = batch['mask_for_losses']

        # L1
        l1_value = masked_l1_loss(predicted_img, img, supervised_mask,
                                  self.config.losses.l1.weight_known,
                                  self.config.losses.l1.weight_missing)

        total_loss = l1_value
        metrics = dict(gen_l1=l1_value)

        # vgg-based perceptual loss
        if self.config.losses.perceptual.weight > 0:
            pl_value = self.loss_pl(predicted_img, img, mask=supervised_mask).sum() * self.config.losses.perceptual.weight
            total_loss = total_loss + pl_value
            metrics['gen_pl'] = pl_value

        # discriminator
        # adversarial_loss calls backward by itself
        mask_for_discr = supervised_mask if self.distance_weighted_mask_for_discr else original_mask
        self.adversarial_loss.pre_generator_step(real_batch=img, fake_batch=predicted_img,
                                                 generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(img)
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(real_batch=img,
                                                                         fake_batch=predicted_img,
                                                                         discr_real_pred=discr_real_pred,
                                                                         discr_fake_pred=discr_fake_pred,
                                                                         mask=mask_for_discr)
        total_loss = total_loss + adv_gen_loss
        metrics['gen_adv'] = adv_gen_loss
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))


        # feature matching
        if self.config.losses.feature_matching.weight > 0:
            need_mask_in_fm = OmegaConf.to_container(self.config.losses.feature_matching).get('pass_mask', False)
            mask_for_fm = supervised_mask if need_mask_in_fm else None
            fm_value = feature_matching_loss(discr_fake_features, discr_real_features,
                                             mask=mask_for_fm) * self.config.losses.feature_matching.weight
            total_loss = total_loss + fm_value
            metrics['gen_fm'] = fm_value

        # # wave_discriminator
        
        if self.config.new_params.wave_dis:
            # adversarial_loss calls backward by itself
            mask_for_wave_discr = supervised_mask if self.distance_weighted_mask_for_discr else original_mask
            self.wave_adversarial_loss.pre_generator_step(real_batch=img, fake_batch=predicted_img,
                                                    generator=self.generator, discriminator=self.wave_discriminator)
            wave_discr_real_pred, wave_discr_real_features = self.wave_discriminator(img)
            wave_discr_fake_pred, wave_discr_fake_features = self.wave_discriminator(predicted_img)
            wave_adv_gen_loss, wave_adv_metrics = self.wave_adversarial_loss.generator_loss(real_batch=img,
                                                                            fake_batch=predicted_img,
                                                                            discr_real_pred=wave_discr_real_pred,
                                                                            discr_fake_pred=wave_discr_fake_pred,
                                                                            mask=mask_for_wave_discr)
            total_loss = total_loss + wave_adv_gen_loss
            metrics['gen_adv_wave'] = wave_adv_gen_loss
            metrics.update(add_prefix_to_keys(wave_adv_metrics, 'adv_wave_'))

            # wave discriminator feature matching
            if self.config.losses.wave_feature_matching.weight > 0:
                need_mask_in_fm = OmegaConf.to_container(self.config.losses.wave_feature_matching).get('pass_mask', False)
                wave_mask_for_fm = supervised_mask if need_mask_in_fm else None
                wave_fm_value = feature_matching_loss(wave_discr_fake_features, wave_discr_real_features,
                                                mask=wave_mask_for_fm) * self.config.losses.wave_feature_matching.weight
                total_loss = total_loss + wave_fm_value
                metrics['gen_fm_wave'] = wave_fm_value
        
        if self.loss_segm_pl is not None:
            segm_pl_value = self.loss_segm_pl(predicted_img, img)
            total_loss = total_loss + segm_pl_value
            metrics['gen_resnet_pl'] = segm_pl_value


        # if self.loss_resnet_pl is not None:
        #     resnet_pl_value = self.loss_resnet_pl(predicted_img, img)
        #     total_loss = total_loss + resnet_pl_value
        #     metrics['gen_resnet_pl'] = resnet_pl_value
        if self.config.new_params.ewc_lambda > 0:
            ewc_loss = self.ewc_loss() * self.config.new_params.ewc_lambda
            total_loss = total_loss + ewc_loss
            metrics['ewc_loss'] = ewc_loss

        if self.training and self.config.new_params.contrast>0:

            contrast_discr_fake_pred, _ = self.contrast_discriminator(predicted_img) 
            current_batch_size = int(contrast_discr_fake_pred.shape[0]/2)
            q = contrast_discr_fake_pred[:current_batch_size].view(current_batch_size, -1)  # q & k shape: (batch_size, 19*19)
            k = contrast_discr_fake_pred[current_batch_size:].view(current_batch_size, -1)
            
            q = torch.nn.functional.normalize(q, dim=1)
            k = torch.nn.functional.normalize(k, dim=1)

            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            # l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone()])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.T

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            # dequeue and enqueue
            self._dequeue_and_enqueue(k)

            contrast_loss = self.criterion(logits, labels) * self.config.new_params.contrast

            total_loss = total_loss + contrast_loss
            metrics['contrast_loss'] = contrast_loss

        if self.training and self.config.new_params.CWD>0:
            preds_S, preds_T = batch['acti_small_model'], batch['acti_model']
            tau = self.config.new_params.temperature
            assert preds_S.shape[-2:] == preds_T.shape[-2:]
            N, C, W, H = preds_S.shape

            softmax_pred_T = F.softmax(preds_T.view(-1, W * H) / tau, dim=1)

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            CWD_loss = torch.sum(softmax_pred_T *
                            logsoftmax(preds_T.view(-1, W * H) / tau) -
                            softmax_pred_T *
                            logsoftmax(preds_S.view(-1, W * H) / tau)) * (
                                tau**2)

            CWD_loss = self.config.new_params.CWD * CWD_loss / (C * N)
            total_loss = total_loss + CWD_loss
            metrics['CWD_loss'] = CWD_loss
        
        if self.config.new_params.ffl > 0:
            ffl_loss = self.config.new_params.ffl * self.loss_ffl(predicted_img, img)
            total_loss = total_loss + ffl_loss
            metrics['ffl_loss'] = ffl_loss

        if self.config.new_params.tv > 0:
            tv_loss = self.total_variation_loss(predicted_img, self.config.new_params.tv)
            total_loss = total_loss + tv_loss
            metrics['tv_loss'] = tv_loss

        # if self.config.new_params.wave_loss > 0:
        #     predicted_image_LL, predicted_image_LH, predicted_image_HL, predicted_image_HH = self.wave_pool(predicted_img)
        #     predicted_H = predicted_image_LH + predicted_image_HL + predicted_image_HH
        #     image_LL, image_LH, image_HL, image_HH = self.wave_pool(img)
        #     image_H = image_LH + image_HL + image_HH
        #     if self.config.new_params.wave_loss_type == 'l1':
        #         if self.config.new_params.wave_loss_frequency == 'four':
        #             r = 0.4
        #             wave_loss = (r*r*F.l1_loss(predicted_image_LL, image_LL, reduction='none') + (1-r)*r*F.l1_loss(predicted_image_LH, image_LH, reduction='none') + 
        #                 (1-r)*r*F.l1_loss(predicted_image_HL, image_HL, reduction='none') + (1-r)*(1-r)*F.l1_loss(predicted_image_HH, image_HH, reduction='none')).mean()
        #         elif self.config.new_params.wave_loss_frequency == 'three':
        #             r = 0.4
        #             wave_loss = ((1-r)*r*F.l1_loss(predicted_image_LH, image_LH, reduction='none') + 
        #                 (1-r)*r*F.l1_loss(predicted_image_HL, image_HL, reduction='none') + (1-r)*(1-r)*F.l1_loss(predicted_image_HH, image_HH, reduction='none')).mean()
        #         elif self.config.new_params.wave_loss_frequency == 'one':
        #             wave_loss = F.l1_loss(predicted_H, image_H, reduction='none')
        #     elif self.config.new_params.wave_loss_type == 'mse':
        #         if self.config.new_params.wave_loss_frequency == 'four':
        #             r = 0.4
        #             wave_loss = (r*r*F.mse_loss(predicted_image_LL, image_LL, reduction='none') + (1-r)*r*F.mse_loss(predicted_image_LH, image_LH, reduction='none') + 
        #                 (1-r)*r*F.mse_loss(predicted_image_HL, image_HL, reduction='none') + (1-r)*(1-r)*F.mse_loss(predicted_image_HH, image_HH, reduction='none')).mean()
        #         elif self.config.new_params.wave_loss_frequency == 'three':
        #             r = 0.4
        #             wave_loss = ((1-r)*r*F.mse_loss(predicted_image_LH, image_LH, reduction='none') + 
        #                 (1-r)*r*F.mse_loss(predicted_image_HL, image_HL, reduction='none') + (1-r)*(1-r)*F.mse_loss(predicted_image_HH, image_HH, reduction='none')).mean()
        #         elif self.config.new_params.wave_loss_frequency == 'one':
        #             wave_loss = F.mse_loss(predicted_H, image_H, reduction='none')
        #     elif self.config.new_params.wave_loss_type == 'segm_pl':
        #         if self.config.new_params.wave_loss_frequency == 'four':
        #             r = 0.4
        #             wave_loss = (r*r*self.loss_segm_pl(predicted_image_LL, image_LL ) + (1-r)*r*self.loss_segm_pl(predicted_image_LH, image_LH ) + 
        #                 (1-r)*r*self.loss_segm_pl(predicted_image_HL, image_HL ) + (1-r)*(1-r)*self.loss_segm_pl(predicted_image_HH, image_HH )).mean()
        #         elif self.config.new_params.wave_loss_frequency == 'three':
        #             r = 0.4
        #             wave_loss = ((1-r)*r*self.loss_segm_pl(predicted_image_LH, image_LH ) + 
        #                 (1-r)*r*self.loss_segm_pl(predicted_image_HL, image_HL ) + (1-r)*(1-r)*self.loss_segm_pl(predicted_image_HH, image_HH )).mean()
        #         elif self.config.new_params.wave_loss_frequency == 'one':
        #             wave_loss = self.loss_segm_pl(predicted_H, image_H )
        #     wave_loss *= self.config.new_params.wave_loss
        #     total_loss = total_loss + wave_loss
        #     metrics['wave_loss'] = wave_loss

        if self.config.new_params.wave_segmpl_loss > 0:
            _, predicted_image_LH, predicted_image_HL, predicted_image_HH = self.wave_pool(predicted_img)
            predicted_H = predicted_image_LH + predicted_image_HL + predicted_image_HH
            _, image_LH, image_HL, image_HH = self.wave_pool(img)
            image_H = image_LH + image_HL + image_HH
            
            wave_segmpl_loss = self.config.new_params.wave_segmpl_loss * self.loss_segm_pl(predicted_H, image_H)
            total_loss = total_loss + wave_segmpl_loss
            metrics['wave_segmpl_loss'] = wave_segmpl_loss

        if self.config.new_params.wave_ffl_loss > 0:
            _, predicted_image_LH, predicted_image_HL, predicted_image_HH = self.wave_pool(predicted_img)
            predicted_H = predicted_image_LH + predicted_image_HL + predicted_image_HH
            _, image_LH, image_HL, image_HH = self.wave_pool(img)
            image_H = image_LH + image_HL + image_HH

            wave_ffl_loss = self.config.new_params.wave_ffl_loss * self.loss_ffl(predicted_H, image_H)
            total_loss = total_loss + wave_ffl_loss
            metrics['wave_ffl_loss'] = wave_ffl_loss

        return total_loss, metrics

    def discriminator_loss(self, batch):
        total_loss = 0
        metrics = {}

        predicted_img = batch[self.image_to_discriminator].detach()
        self.adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=predicted_img,
                                                     generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(batch['image'])
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        adv_discr_loss, adv_metrics = self.adversarial_loss.discriminator_loss(real_batch=batch['image'],
                                                                               fake_batch=predicted_img,
                                                                               discr_real_pred=discr_real_pred,
                                                                               discr_fake_pred=discr_fake_pred,
                                                                               mask=batch['mask'])
        total_loss = total_loss + adv_discr_loss
        metrics['discr_adv'] = adv_discr_loss
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))


        if batch.get('use_fake_fakes', False):
            fake_fakes = batch['fake_fakes']
            self.adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=fake_fakes,
                                                         generator=self.generator, discriminator=self.discriminator)
            discr_fake_fakes_pred, _ = self.discriminator(fake_fakes)
            fake_fakes_adv_discr_loss, fake_fakes_adv_metrics = self.adversarial_loss.discriminator_loss(
                real_batch=batch['image'],
                fake_batch=fake_fakes,
                discr_real_pred=discr_real_pred,
                discr_fake_pred=discr_fake_fakes_pred,
                mask=batch['mask']
            )
            total_loss = total_loss + fake_fakes_adv_discr_loss
            metrics['discr_adv_fake_fakes'] = fake_fakes_adv_discr_loss
            metrics.update(add_prefix_to_keys(fake_fakes_adv_metrics, 'adv_'))

        
        if self.config.new_params.wave_dis:
            self.wave_adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=predicted_img,
                                                        generator=self.generator, discriminator=self.wave_discriminator)
            wave_discr_real_pred, discr_real_features = self.wave_discriminator(batch['image'])
            wave_discr_fake_pred, discr_fake_features = self.wave_discriminator(predicted_img)
            wave_adv_discr_loss, wave_adv_metrics = self.wave_adversarial_loss.discriminator_loss(real_batch=batch['image'],
                                                                                fake_batch=predicted_img,
                                                                                discr_real_pred=wave_discr_real_pred,
                                                                                discr_fake_pred=wave_discr_fake_pred,
                                                                                mask=batch['mask'])
            total_loss = total_loss + wave_adv_discr_loss
            metrics['discr_adv_wave'] = wave_adv_discr_loss
            metrics.update(add_prefix_to_keys(wave_adv_metrics, 'adv_wave_'))


            if batch.get('use_fake_fakes', False):
                fake_fakes = batch['fake_fakes']
                self.wave_adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=fake_fakes,
                                                            generator=self.generator, discriminator=self.wave_discriminator)
                wave_discr_fake_fakes_pred, _ = self.wave_discriminator(fake_fakes)
                wave_fake_fakes_adv_discr_loss, wave_fake_fakes_adv_metrics = self.wave_adversarial_loss.discriminator_loss(
                    real_batch=batch['image'],
                    fake_batch=fake_fakes,
                    discr_real_pred=wave_discr_real_pred,
                    discr_fake_pred=wave_discr_fake_fakes_pred,
                    mask=batch['mask']
                )
                total_loss = total_loss + wave_fake_fakes_adv_discr_loss
                metrics['discr_adv_wave_fake_fakes'] = wave_fake_fakes_adv_discr_loss
                metrics.update(add_prefix_to_keys(wave_fake_fakes_adv_metrics, 'adv_wave_'))

        return total_loss, metrics

    def total_variation_loss(self, img, weight):
        bs_img, c_img, h_img, w_img = img.size()
        tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
        return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # # gather keys before updating queue
        # keys = self.concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
    
    # # utils
    # @torch.no_grad()
    # def concat_all_gather(tensor):
    #     """
    #     Performs all_gather operation on the provided tensors.
    #     *** Warning ***: torch.distributed.all_gather has no gradient.
    #     """
    #     tensors_gather = [torch.ones_like(tensor)
    #         for _ in range(torch.distributed.get_world_size())]
    #     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    #     output = torch.cat(tensors_gather, dim=0)
    #     return output

    def estimate_fisher(self, sample_size=1024, batch_size=16):
        # sample loglikelihoods from the dataset.
        data_config_for_ewc = self.config.data.train
        data_config_for_ewc.indir = '/home/mona/codes/lama/datasets/val256crop'
        data_loader = make_default_train_dataloader(**data_config_for_ewc)

        self.generator.to('cuda:0')
        self.discriminator.to('cuda:0')
        
        for optimizer in self.optimizers():
            optimizer.zero_grad()

        # loglikelihoods = []


        set_requires_grad(self.generator, True)
        set_requires_grad(self.random_generator, False)
        set_requires_grad(self.discriminator, False)
        count_data = 0
        for batch in data_loader:
            # print(11111)
            img = Variable(batch['image']).cuda()
            mask = Variable(batch['mask']).cuda()
            # img = batch['image']
            # mask = batch['mask']
            masked_img = img * (1 - mask)

            if self.concat_mask:
                masked_img = torch.cat([masked_img, mask], dim=1)
            # masked_img = masked_img.to('cuda:0')
            predicted_image = self.generator(masked_img)

            discr_fake_pred, _ = self.discriminator(predicted_image)
            fake_loss = F.softplus(-discr_fake_pred)
            # fake_loss = fake_loss.mean(dim=[2, 3])
            fake_loss = fake_loss.mean()
            fake_loss.backward()
            # self.manual_backward(fake_loss)

            # loglikelihoods.append( fake_loss )
            
            count_data += 1
            if count_data >= sample_size // batch_size:
                break
        # estimate the fisher information of the parameters.
        # loglikelihoods = torch.cat(loglikelihoods).unbind()
        # loglikelihood_grads = zip(*[autograd.grad(
        #     l, self.parameters(),
        #     retain_graph=(i < len(loglikelihoods))
        # ) for i, l in enumerate(loglikelihoods, 1)])
        # loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
        # fisher_diagonals = [(g ** 2).mean(0) for g in loglikelihood_grads]
        param_names = []
        fisher_diagonals = []
        # param_names = [
        #     n.replace('.', '__') for n, p in self.named_parameters()
        # ]
        fisher_layer = {}
        for name, param in self.named_parameters():
            if 'generator.model' in name and '.weight' in name and 'bn' not in name and param.dim()!=1:
            # if 'generator.model' in name and '.weight' in name and 'bn' not in name and param.dim()!=1 and '.conv1.' not in name and '.conv2.' not in name:
    
                if self.config.new_params.ewc_onlyFFCResnet:
                    pattern = r'\.[0-9]*\.'
                    layer_num = re.search(pattern, name).group()[1:-1]
                    if int(layer_num)<5 or int(layer_num)>22:
                        continue
                if self.config.new_params.ewc_onlyUp:
                    if 'random_generator' in name:
                        continue
                    pattern = r'\.[0-9]*\.'
                    layer_num = re.search(pattern, name).group()[1:-1]
                    
                    if int(layer_num)<23:
                        continue
                param_names.append(name.replace('.', '_'))
                fisher_diagonals.append(param.grad.data.clone().pow(2)) 

        #         pattern = r'\.[0-9]*\.'
        #         layer_num = re.search(pattern, name).group()[1:-1]

        #         if int(layer_num)<5 or int(layer_num)>22:
        #             continue

        #         if fisher_layer.get(layer_num)==None:
        #             fisher_layer[layer_num] = [fisher_diagonals[-1].mean().item()]
        #         else:
        #             fisher_layer[layer_num].append(fisher_diagonals[-1].mean().item())


        # xx = []
        # yy = []
        # for k, v in fisher_layer.items():
        # # print(type(k), type(v[0]))
        #     xx.append('{}'.format(k))
        #     yy.append(mean(v))
        #     print('the average Fisher information of weights at block {}'.format(k) + ': ' + str(mean(v)))
        # # print(param_names)
        # plt.bar(xx, yy, align =  'center') 
        # plt.title('Fisher information')
        # # plt.ylabel('Fisher information') 
        # # plt.show()
        # plt.savefig('/home/mona/codes/lama/fisher_sample{}_onlyFFCResnet.png'.format(sample_size))
        # # plt.xlabel('X axis') 
        # exit(0)
        if self.config.new_params.ewc_onlyUp:
            for params, random_params in zip(self.generator.model[23:].parameters(), self.random_generator.model[23:].parameters()):
                params.data.copy_(random_params.data)  # initialize
        return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}

    def consolidate(self, fisher):
        for n, p in self.named_parameters():
            if 'generator.model' in n and '.weight' in n and 'bn' not in n and p.dim()!=1:
            # if 'generator.model' in n and '.weight' in n and 'bn' not in n and p.dim()!=1 and '.conv1.' not in n and '.conv2.' not in n:
                if self.config.new_params.ewc_onlyFFCResnet:
                    pattern = r'\.[0-9]*\.'
                    layer_num = re.search(pattern, n).group()[1:-1]
                    if int(layer_num)<5 or int(layer_num)>22:
                        continue
                if self.config.new_params.ewc_onlyUp:
                    if 'random_generator' in n:
                        continue
                    pattern = r'\.[0-9]*\.'
                    layer_num = re.search(pattern, n).group()[1:-1]
                    if int(layer_num)<23:
                        continue
                n = n.replace('.', '_')
                self.register_buffer('{}_mean'.format(n), p.data.clone())
                self.register_buffer('{}_fisher'
                                    .format(n), fisher[n].data.clone())
    def ewc_loss(self, cuda=False):
        try:
            losses = []
            for n, p in self.named_parameters():
                if 'generator.model' in n and '.weight' in n and 'bn' not in n and p.dim()!=1:
                # if 'generator.model' in n and '.weight' in n and 'bn' not in n and p.dim()!=1 and '.conv1.' not in n and '.conv2.' not in n:
                    # retrieve the consolidated mean and fisher information.
                    if self.config.new_params.ewc_onlyFFCResnet:
                        pattern = r'\.[0-9]*\.'
                        layer_num = re.search(pattern, n).group()[1:-1]
                        if int(layer_num)<5 or int(layer_num)>22:
                            continue
                    if self.config.new_params.ewc_onlyUp:
                        if 'random_generator' in n:
                            continue
                        pattern = r'\.[0-9]*\.'
                        layer_num = re.search(pattern, n).group()[1:-1]
                        if int(layer_num)<23:
                            continue
                    n = n.replace('.', '_')
                    mean = getattr(self, '{}_mean'.format(n))
                    fisher = getattr(self, '{}_fisher'.format(n))
                    # wrap mean and fisher in variables.
                    mean = Variable(mean)
                    fisher = Variable(fisher)
                    # calculate a ewc loss. (assumes the parameter's prior as
                    # gaussian distribution with the estimated mean and the
                    # estimated cramer-rao lower bound variance, which is
                    # equivalent to the inverse of fisher information)
                    losses.append((fisher * (p-mean)**2).sum())
            return sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )


