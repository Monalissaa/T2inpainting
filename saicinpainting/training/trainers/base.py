import copy
import logging
from typing import Dict, Tuple

import pandas as pd
import pytorch_lightning as ptl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DistributedSampler

from saicinpainting.evaluation import make_evaluator
from saicinpainting.training.data.datasets import make_default_train_dataloader, make_default_val_dataloader
from saicinpainting.training.losses.adversarial import make_discrim_loss
from saicinpainting.training.losses.perceptual import PerceptualLoss, ResNetPL
from saicinpainting.training.modules import make_generator, make_discriminator
from saicinpainting.training.visualizers import make_visualizer
from saicinpainting.utils import add_prefix_to_keys, average_dicts, set_requires_grad, set_requires_grad_freezeD, flatten_dict, \
    get_has_ddp_rank

from saicinpainting.training.modules import agent_net

from focal_frequency_loss import FocalFrequencyLoss as FFL

from saicinpainting.training.modules.wave import WavePool

from saicinpainting.training.visualizers.visualizer import HtmlPageVisualizer, load_image 
import os


LOGGER = logging.getLogger(__name__)


def make_optimizer(parameters, kind='adamw', **kwargs):
    if kind == 'adam':
        optimizer_class = torch.optim.Adam
    elif kind == 'adamw':
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(f'Unknown optimizer kind {kind}')
    return optimizer_class(parameters, **kwargs)


def update_running_average(result: nn.Module, new_iterate_model: nn.Module, decay=0.999):
    with torch.no_grad():
        res_params = dict(result.named_parameters())
        new_params = dict(new_iterate_model.named_parameters())

        for k in res_params.keys():
            res_params[k].data.mul_(decay).add_(new_params[k].data, alpha=1 - decay)


def make_multiscale_noise(base_tensor, scales=6, scale_mode='bilinear'):
    batch_size, _, height, width = base_tensor.shape
    cur_height, cur_width = height, width
    result = []
    align_corners = False if scale_mode in ('bilinear', 'bicubic') else None
    for _ in range(scales):
        cur_sample = torch.randn(batch_size, 1, cur_height, cur_width, device=base_tensor.device)
        cur_sample_scaled = F.interpolate(cur_sample, size=(height, width), mode=scale_mode, align_corners=align_corners)
        result.append(cur_sample_scaled)
        cur_height //= 2
        cur_width //= 2
    return torch.cat(result, dim=1)


class BaseInpaintingTrainingModule(ptl.LightningModule):
    def __init__(self, config, use_ddp, *args,  predict_only=False, visualize_each_iters=100,
                 average_generator=False, generator_avg_beta=0.999, average_generator_start_step=30000,
                 average_generator_period=10, store_discr_outputs_for_vis=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        LOGGER.info('BaseInpaintingTrainingModule init called')

        self.config = config

        if config.new_params.spottune:
            self.agent = agent_net.resnet(self.config.generator.n_blocks * 2)

        self.generator = make_generator(config, **self.config.generator)
        if self.config.new_params.ewc_onlyUp:
            self.random_generator = make_generator(config, **self.config.generator)
        self.use_ddp = use_ddp

        if not get_has_ddp_rank():
            LOGGER.info(f'Generator\n{self.generator}')

        if not predict_only:
            self.save_hyperparameters(self.config)
            self.discriminator = make_discriminator(**self.config.discriminator)
            if self.config.new_params.only_pl_loss==False:
                self.adversarial_loss = make_discrim_loss(**self.config.losses.adversarial)
            
            self.visualizer = make_visualizer(**self.config.visualizer)
            self.visualizer_html = HtmlPageVisualizer(num_rows = 500, num_cols = 1)
            self.val_evaluator = make_evaluator(**self.config.evaluator)
            self.test_evaluator = make_evaluator(**self.config.evaluator)

            if self.config.new_params.wave_dis:
                self.wave_adversarial_loss = make_discrim_loss(**self.config.losses.wave_adversarial)
                self.wave_discriminator = make_discriminator(**self.config.wave_discriminator)

            if self.config.new_params.contrast>0: # same architecture as origin discriminator
                self.contrast_discriminator = make_discriminator(**self.config.discriminator)
                self.K = 88 # K = int(100/8) * 8
                self.T = 0.07 # same as moco
                # create the queue
                self.register_buffer("queue", torch.randn(19*19, self.K))
                self.queue = nn.functional.normalize(self.queue, dim=0)

                self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

                # define loss function (criterion) and optimizer
                self.criterion = nn.CrossEntropyLoss().cuda()

            if not get_has_ddp_rank():
                LOGGER.info(f'Discriminator\n{self.discriminator}')

            extra_val = self.config.data.get('extra_val', ())
            if extra_val:
                self.extra_val_titles = list(extra_val)
                self.extra_evaluators = nn.ModuleDict({k: make_evaluator(**self.config.evaluator)
                                                       for k in extra_val})
            else:
                self.extra_evaluators = {}

            self.average_generator = average_generator
            self.generator_avg_beta = generator_avg_beta
            self.average_generator_start_step = average_generator_start_step
            self.average_generator_period = average_generator_period
            self.generator_average = None
            self.last_generator_averaging_step = -1
            self.store_discr_outputs_for_vis = store_discr_outputs_for_vis

            if self.config.losses.get("l1", {"weight_known": 0})['weight_known'] > 0:
                self.loss_l1 = nn.L1Loss(reduction='none')

            if self.config.losses.get("mse", {"weight": 0})['weight'] > 0:
                self.loss_mse = nn.MSELoss(reduction='none')
            
            if self.config.losses.perceptual.weight > 0:
                self.loss_pl = PerceptualLoss()

            if self.config.losses.get("segm_pl", {"weight": 0})['weight'] > 0:
                self.loss_segm_pl = ResNetPL(**self.config.losses.segm_pl)
            else:
                self.loss_segm_pl = None

            if self.config.new_params.ffl > 0  or self.config.new_params.wave_ffl_loss > 0:
                self.loss_ffl = FFL(loss_weight=1.0, alpha=1.0)

            if self.config.new_params.wave_ffl_loss > 0 or self.config.new_params.wave_segmpl_loss > 0:
                self.wave_pool = WavePool(3)

            # if self.config.losses.get("resnet_pl", {"weight": 0})['weight'] > 0:
            #     self.loss_resnet_pl = ResNetPL(**self.config.losses.resnet_pl)
            # else:
            #     self.loss_resnet_pl = None

            total_epoch_number = self.config.trainer.kwargs.max_epochs
            self.first_stage_epochs = int(total_epoch_number/3)
            self.second_stage_epochs = self.first_stage_epochs * 2
            self.jump_number = self.config.new_params.jump_number  # Number of epochs between two jumps

        self.visualize_each_iters = visualize_each_iters
        LOGGER.info('BaseInpaintingTrainingModule init done')

        self.fisher_information_flag = False
        self.weight_decay_only_conv_weight_flag = False
        if self.config.new_params.ewc_lambda > 0:
            self.fisher_information_flag = True
        if self.config.new_params.weight_decay_only_conv_weight_lambda > 0:
            self.weight_decay_only_conv_weight_flag = True


    def configure_optimizers(self):
        if self.config.new_params.only_pl_loss==False:
            discriminator_params = list(self.discriminator.parameters())
        if self.config.new_params.spottune:
            return [
                dict(optimizer=make_optimizer(self.generator.parameters(), **self.config.optimizers.generator)),
                dict(optimizer=make_optimizer(discriminator_params, **self.config.optimizers.discriminator)),
                dict(optimizer=make_optimizer(self.agent.parameters(), **self.config.optimizers.discriminator)),
            ]
        elif self.config.new_params.wave_dis:
            wave_discriminator_params = list(self.wave_discriminator.parameters())
            return [
                dict(optimizer=make_optimizer(self.generator.parameters(), **self.config.optimizers.generator)),
                dict(optimizer=make_optimizer(discriminator_params, **self.config.optimizers.discriminator)),
                dict(optimizer=make_optimizer(wave_discriminator_params, **self.config.optimizers.discriminator)),
            ]
        elif self.config.new_params.contrast>0:
            return [
                dict(optimizer=make_optimizer(self.generator.parameters(), **self.config.optimizers.generator)),
                dict(optimizer=make_optimizer(discriminator_params, **self.config.optimizers.discriminator)),
                dict(optimizer=make_optimizer(self.contrast_discriminator.parameters(), **self.config.optimizers.discriminator)),
            ]
        elif self.config.new_params.CWD>0:
            return [
                dict(optimizer=make_optimizer(self.generator.small_model.parameters(), **self.config.optimizers.generator)),
                dict(optimizer=make_optimizer(discriminator_params, **self.config.optimizers.discriminator)),
            ]
        elif self.config.new_params.only_pl_loss:
            return [
                dict(optimizer=make_optimizer(self.generator.parameters(), **self.config.optimizers.generator)),
            ]
        elif self.config.new_params.fix_add_ffc:
            return [
                dict(optimizer=make_optimizer(self.generator.additional_model.parameters(), **self.config.optimizers.generator)),
                dict(optimizer=make_optimizer(discriminator_params, **self.config.optimizers.discriminator)),
            ] 
        else:
            return [
                dict(optimizer=make_optimizer(self.generator.parameters(), **self.config.optimizers.generator)),
                dict(optimizer=make_optimizer(discriminator_params, **self.config.optimizers.discriminator)),
            ]

    def train_dataloader(self):
        kwargs = dict(self.config.data.train)
        if self.use_ddp:
            kwargs['ddp_kwargs'] = dict(num_replicas=self.trainer.num_nodes * self.trainer.num_processes,
                                        rank=self.trainer.global_rank,
                                        shuffle=True)
        if self.config.new_params.contrast > 0:
            dataloader = make_default_train_dataloader(**self.config.data.train, kind='contrast')
        else:
            dataloader = make_default_train_dataloader(**self.config.data.train)
        return dataloader

    def val_dataloader(self):
        res = [make_default_val_dataloader(**self.config.data.val)]

        if self.config.data.visual_test is not None:
            res = res + [make_default_val_dataloader(**self.config.data.visual_test)]
        else:
            res = res + res

        extra_val = self.config.data.get('extra_val', ())
        if extra_val:
            res += [make_default_val_dataloader(**extra_val[k]) for k in self.extra_val_titles]

        return res

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        self._is_training_step = True
        return self._do_step(batch, batch_idx, mode='train', optimizer_idx=optimizer_idx)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        extra_val_key = None
        if dataloader_idx == 0:
            mode = 'val'
        elif dataloader_idx == 1:
            mode = 'test'
        else:
            mode = 'extra_val'
            extra_val_key = self.extra_val_titles[dataloader_idx - 2]
        self._is_training_step = False
        return self._do_step(batch, batch_idx, mode=mode, extra_val_key=extra_val_key)

    def training_step_end(self, batch_parts_outputs):
        if self.training and self.average_generator \
                and self.global_step >= self.average_generator_start_step \
                and self.global_step >= self.last_generator_averaging_step + self.average_generator_period:
            if self.generator_average is None:
                self.generator_average = copy.deepcopy(self.generator)
            else:
                update_running_average(self.generator_average, self.generator, decay=self.generator_avg_beta)
            self.last_generator_averaging_step = self.global_step

        full_loss = (batch_parts_outputs['loss'].mean()
                     if torch.is_tensor(batch_parts_outputs['loss'])  # loss is not tensor when no discriminator used
                     else torch.tensor(batch_parts_outputs['loss']).float().requires_grad_(True))
        log_info = {k: v.mean() for k, v in batch_parts_outputs['log_info'].items()}
        self.log_dict(log_info, on_step=True, on_epoch=False)
        return full_loss

    def validation_epoch_end(self, outputs):
        vis_suffix = '_test'
        curoutdir = os.path.join(self.config.visualizer.outdir, f'epoch{self.current_epoch:04d}{vis_suffix}')
        self.visualizer_html.save(f'{curoutdir}/val_results.html')

        outputs = [step_out for out_group in outputs for step_out in out_group]
        averaged_logs = average_dicts(step_out['log_info'] for step_out in outputs)
        self.log_dict({k: v.mean() for k, v in averaged_logs.items()})

        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        # standard validation
        val_evaluator_states = [s['val_evaluator_state'] for s in outputs if 'val_evaluator_state' in s]
        val_evaluator_res = self.val_evaluator.evaluation_end(states=val_evaluator_states)
        val_evaluator_res_df = pd.DataFrame(val_evaluator_res).stack(1).unstack(0)
        val_evaluator_res_df.dropna(axis=1, how='all', inplace=True)
        LOGGER.info(f'Validation metrics after epoch #{self.current_epoch}, '
                    f'total {self.global_step} iterations:\n{val_evaluator_res_df}')

        for k, v in flatten_dict(val_evaluator_res).items():
            self.log(f'val_{k}', v)

        # standard visual test
        # test_evaluator_states = [s['test_evaluator_state'] for s in outputs
        #                          if 'test_evaluator_state' in s]
        # test_evaluator_res = self.test_evaluator.evaluation_end(states=test_evaluator_states)
        # test_evaluator_res_df = pd.DataFrame(test_evaluator_res).stack(1).unstack(0)
        # test_evaluator_res_df.dropna(axis=1, how='all', inplace=True)
        # LOGGER.info(f'Test metrics after epoch #{self.current_epoch}, '
        #             f'total {self.global_step} iterations:\n{test_evaluator_res_df}')
        #
        # for k, v in flatten_dict(test_evaluator_res).items():
        #     self.log(f'test_{k}', v)

        # extra validations
        if self.extra_evaluators:
            for cur_eval_title, cur_evaluator in self.extra_evaluators.items():
                cur_state_key = f'extra_val_{cur_eval_title}_evaluator_state'
                cur_states = [s[cur_state_key] for s in outputs if cur_state_key in s]
                cur_evaluator_res = cur_evaluator.evaluation_end(states=cur_states)
                cur_evaluator_res_df = pd.DataFrame(cur_evaluator_res).stack(1).unstack(0)
                cur_evaluator_res_df.dropna(axis=1, how='all', inplace=True)
                LOGGER.info(f'Extra val {cur_eval_title} metrics after epoch #{self.current_epoch}, '
                            f'total {self.global_step} iterations:\n{cur_evaluator_res_df}')
                for k, v in flatten_dict(cur_evaluator_res).items():
                    self.log(f'extra_val_{cur_eval_title}_{k}', v)

    def _do_step(self, batch, batch_idx, mode='train', optimizer_idx=None, extra_val_key=None):
        # torch.autograd.set_detect_anomaly(True)
        if self.fisher_information_flag and self.config.new_params.ewc_lambda > 0:
            print('compute the fisher information!!!')
            self.consolidate(self.estimate_fisher())
            self.fisher_information_flag = False
            # self.config.new_params.ewc_lambda = None
            print('Done!!!')

        if self.weight_decay_only_conv_weight_flag and self.config.new_params.weight_decay_only_conv_weight_lambda > 0:
            print('#########################record origin params data########################')
            self.record_origin_params_data()
            self.weight_decay_only_conv_weight_flag = False


        if self.config.new_params.two_stage:
            if mode=='train':
                if self.current_epoch < self.first_stage_epochs:
                    self.generator.alpha = 0
                elif self.current_epoch >= self.first_stage_epochs and self.current_epoch < self.second_stage_epochs:
                    if self.current_epoch % self.jump_number == 0:
                        self.generator.alpha = (self.current_epoch-self.first_stage_epochs)/self.first_stage_epochs
                    if self.generator.alpha == 0:
                        self.generator.alpha = 0.001 # for valid gradient
                else:
                    self.generator.alpha = 1
            else:
                self.generator.alpha = 1

            if optimizer_idx == 0:  # step for generator
                set_requires_grad(self.generator, False)
                set_requires_grad(self.discriminator, False)
                if self.generator.alpha == 0:
                    set_requires_grad(self.generator.fs_model, True)
                else:
                    # set_requires_grad(self.generator.model, True)
                    feature_loc = self.config.new_params.freeze_blocks + 1
                    for loc in range(feature_loc, 26):
                        set_requires_grad_freezeD(self.generator.model, True, target_layer=f'{loc}.')
            elif optimizer_idx == 1:  # step for discriminator
                set_requires_grad(self.generator, False)
                set_requires_grad(self.discriminator, True)

        else:
            if optimizer_idx == 0:  # step for generator
                set_requires_grad(self.generator, False)
                set_requires_grad(self.discriminator, False)

                
                # if self.config.new_params.wave_dis:
                #     set_requires_grad(self.wave_discriminator, False)
                # if self.config.new_params.contrast>0:
                #     set_requires_grad(self.contrast_discriminator, True)



                if self.config.new_params.fix:
                    feature_loc = self.config.new_params.feature_loc
                    for loc in feature_loc:
                        set_requires_grad_freezeD(self.generator, True, target_layer=f'model.{loc}.')
                    
                elif self.config.new_params.fix_add_ffc:
                    set_requires_grad(self.generator.additional_model, True)
                elif self.config.new_params.two_stage_from_init==False and (self.config.new_params.tsa.four or self.config.new_params.tsa.two or self.config.new_params.tsa.one \
                    or self.config.new_params.tsa.g2g or self.config.new_params.tsa.g2g_down_up):
                    # for name, _ in self.generator.named_parameters():
                    #     if 'alpha' in name:
                    set_requires_grad_freezeD(self.generator, True, target_layer='alpha')
                    if self.config.new_params.tsa.release_tsa_global_bn:
                        set_requires_grad_freezeD(self.generator, True, target_layer='bn')
                        set_requires_grad_freezeD(self.generator, True, target_layer='convl2g')
                        set_requires_grad_freezeD(self.generator, True, target_layer='convg2g')
                        for loc in [25,28,31]:
                            set_requires_grad_freezeD(self.generator, True, target_layer=f'model.{loc}.')
                    elif self.config.new_params.tsa.release_tsa_global:
                        set_requires_grad_freezeD(self.generator, True, target_layer='convl2g')
                        set_requires_grad_freezeD(self.generator, True, target_layer='convg2g')
                    # ----------- next ---------------
                    if self.config.new_params.tsa.end_to_end:
                        if self.current_epoch>=(self.config.trainer.kwargs.max_epochs/2):
                            set_requires_grad(self.generator.model, True)
                            set_requires_grad_freezeD(self.generator, False, target_layer='alpha')

                    # --------------------------------
                    # if self.config.new_params.tsa.pa:
                    #     set_requires_grad_freezeD(self.generator, True, target_layer='pa')
                    # if self.config.new_params.tsa.first_block:
                    #     set_requires_grad_freezeD(self.generator, True, target_layer='model.1.')
                    # if self.config.new_params.tsa.last_block:
                    #     set_requires_grad_freezeD(self.generator, True, target_layer='model.34.')   
                    # if self.config.new_params.tsa.UpDown_blocks: 
                    #     for loc in range(1,5):
                    #         set_requires_grad_freezeD(self.generator, True, target_layer=f'model.{loc}.')
                    #     for loc in range(24, 36):
                    #         set_requires_grad_freezeD(self.generator, True, target_layer=f'model.{loc}.')
                else:
                    set_requires_grad(self.generator.model, True)
                    if self.config.new_params.spottune:
                        set_requires_grad(self.agent, True)
                    elif self.config.new_params.tsa.fix_alpha:
                        set_requires_grad_freezeD(self.generator, False, target_layer='alpha')

                if self.config.new_params.only_bn:
                    for name, _ in self.generator.named_parameters():
                        if 'bn' not in name:
                            set_requires_grad_freezeD(self.generator, False, target_layer=name)
                elif self.config.new_params.only_global_bn:
                    for name, _ in self.generator.named_parameters():
                        if ('bn' not in name) or ('bn_l' in name):
                            set_requires_grad_freezeD(self.generator, False, target_layer=name)
                elif self.config.new_params.only_g2g_bn:
                    for name, _ in self.generator.named_parameters():
                        if ('bn' not in name) or ('bn_l' in name) or ('bn_g' in name):
                            set_requires_grad_freezeD(self.generator, False, target_layer=name)

                if self.config.new_params.release_bn:
                    feature_loc_bn = self.config.new_params.bn_number
                    for loc in feature_loc_bn:
                        set_requires_grad_freezeD(self.generator, True, target_layer=f'model.{loc}.bn')

                elif self.config.new_params.CWD>0:
                    feature_loc =  [x+5 for x in range(self.config.generator.n_blocks_small)]
                    for loc in feature_loc:
                        set_requires_grad_freezeD(self.generator, True, target_layer=f'small_model.{loc}.')
                elif self.config.new_params.fix_middleBlocks_convl2l_convg2l:
                    set_requires_grad_freezeD(self.generator, False, target_layer=f'conv1.ffc.convl2l')
                    set_requires_grad_freezeD(self.generator, False, target_layer=f'conv1.ffc.convg2l')
                    set_requires_grad_freezeD(self.generator, False, target_layer=f'conv2.ffc.convl2l')
                    set_requires_grad_freezeD(self.generator, False, target_layer=f'conv2.ffc.convg2l')
                    
                elif self.config.new_params.fix_middleBlocks_convl2g_convg2g:
                    set_requires_grad_freezeD(self.generator, False, target_layer=f'conv1.ffc.convl2g')
                    set_requires_grad_freezeD(self.generator, False, target_layer=f'conv1.ffc.convg2g')
                    set_requires_grad_freezeD(self.generator, False, target_layer=f'conv2.ffc.convl2g')
                    set_requires_grad_freezeD(self.generator, False, target_layer=f'conv2.ffc.convg2g')
                elif self.config.new_params.fix_middleBlocks_convl2l_convg2l_convl2g:
                    set_requires_grad_freezeD(self.generator, False, target_layer=f'conv1.ffc.convl2l')
                    set_requires_grad_freezeD(self.generator, False, target_layer=f'conv1.ffc.convg2l')
                    set_requires_grad_freezeD(self.generator, False, target_layer=f'conv1.ffc.convl2g')
                    set_requires_grad_freezeD(self.generator, False, target_layer=f'conv2.ffc.convl2l')
                    set_requires_grad_freezeD(self.generator, False, target_layer=f'conv2.ffc.convg2l')
                    set_requires_grad_freezeD(self.generator, False, target_layer=f'conv2.ffc.convl2g')

        
                
                # for name, param in self.generator.named_parameters():
                #         print(name + '_requires_grad: ' + str(param.requires_grad))
                # exit(0)

                # set_requires_grad(self.wave_discriminator, False) 
                # print('for generator')
                # if self.current_epoch==1:
                #     for name, param in self.generator.named_parameters():
                #         print(name + '_requires_grad: ' + str(param.requires_grad))
                # if self.current_epoch==11:
                #     for name, param in self.generator.named_parameters():
                #         print(name + '_requires_grad: ' + str(param.requires_grad))
                #     exit(0)
            elif optimizer_idx == 1:  # step for discriminator
                set_requires_grad(self.generator, False)
                
                if self.config.new_params.spottune:
                        set_requires_grad(self.agent, False)

                if self.config.new_params.contrast>0:
                    set_requires_grad(self.contrast_discriminator, False)

                set_requires_grad(self.discriminator, True)
                if self.config.new_params.wave_dis:
                    set_requires_grad_freezeD(self.wave_discriminator, True, target_layer=f'model')
            
                if self.config.new_params.freezeD>0:
                    for loc in range(int(self.config.new_params.freezeD)):
                        set_requires_grad_freezeD(self.discriminator, False, target_layer=f'model{loc}.')
                # for name, param in self.discriminator.named_parameters():
                #     print(name + '_requires_grad: ' + str(param.requires_grad))
                # exit(0)
                # set_requires_grad(self.wave_discriminator.model, True)
                # set_requires_grad_freezeD(self.wave_discriminator, True, target_layer=f'model')
                # print wave_discriminator --> check whether wavepool is require grad or not

                # feature_loc = 5
                # for loc in range(feature_loc):
                #     set_requires_grad_freezeD(self.discriminator, True, target_layer=f'model{5 - loc}')
                # print('for wave_discriminator')
                # for name, param in self.wave_discriminator.named_parameters():
                #     print(name + '_requires_grad: ' + str(param.requires_grad))
                # exit(0)

                # model0.0.weight
                # model0.0.bias
                # model1.0.weight
                # model1.0.bias
                # model1.1.weight
                # model1.1.bias
                # model2.0.weight
                # model2.0.bias
                # model2.1.weight
                # model2.1.bias
                # model3.0.weight
                # model3.0.bias
                # model3.1.weight
                # model3.1.bias
                # model4.0.weight
                # model4.0.bias
                # model4.1.weight
                # model4.1.bias
                # model5.0.weight
                # model5.0.bias


        batch = self(batch, optimizer_idx)

        total_loss = 0
        metrics = {}

        if optimizer_idx is None or optimizer_idx == 0:  # step for generator
            total_loss, metrics = self.generator_loss(batch)

        elif optimizer_idx is None or optimizer_idx == 1:  # step for discriminator
            if self.config.losses.adversarial.weight > 0:
                total_loss, metrics = self.discriminator_loss(batch)

        if self.get_ddp_rank() in (None, 0) and (batch_idx % self.visualize_each_iters == 0 or mode == 'test'):
            if self.config.losses.adversarial.weight > 0:
                if self.store_discr_outputs_for_vis:
                    with torch.no_grad():
                        self.store_discr_outputs(batch)
            vis_suffix = f'_{mode}'
            if mode == 'extra_val':
                vis_suffix += f'_{extra_val_key}'
            if mode == 'test':
                self.visualizer(self.current_epoch, batch_idx, batch, suffix=vis_suffix, visualizer_html=self.visualizer_html)
            else:
                self.visualizer(self.current_epoch, batch_idx, batch, suffix=vis_suffix)
            # self.visualizer(self.current_epoch, batch_idx, batch, suffix=vis_suffix)

        metrics_prefix = f'{mode}_'
        if mode == 'extra_val':
            metrics_prefix += f'{extra_val_key}_'
        result = dict(loss=total_loss, log_info=add_prefix_to_keys(metrics, metrics_prefix))
        if mode == 'val':
            result['val_evaluator_state'] = self.val_evaluator.process_batch(batch)
        elif mode == 'test':
            result['test_evaluator_state'] = self.test_evaluator.process_batch(batch)
        elif mode == 'extra_val':
            result[f'extra_val_{extra_val_key}_evaluator_state'] = self.extra_evaluators[extra_val_key].process_batch(batch)

        return result

    def get_current_generator(self, no_average=False):
        if not no_average and not self.training and self.average_generator and self.generator_average is not None:
            return self.generator_average
        return self.generator

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Pass data through generator and obtain at leas 'predicted_image' and 'inpainted' keys"""
        raise NotImplementedError()

    def generator_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def discriminator_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def store_discr_outputs(self, batch):
        out_size = batch['image'].shape[2:]
        discr_real_out, _ = self.discriminator(batch['image'])
        discr_fake_out, _ = self.discriminator(batch['predicted_image'])
        batch['discr_output_real'] = F.interpolate(discr_real_out, size=out_size, mode='nearest')
        batch['discr_output_fake'] = F.interpolate(discr_fake_out, size=out_size, mode='nearest')
        batch['discr_output_diff'] = batch['discr_output_real'] - batch['discr_output_fake']
        if self.config.new_params.wave_dis:
            wave_discr_real_out, _ = self.wave_discriminator(batch['image'])
            wave_discr_fake_out, _ = self.wave_discriminator(batch['predicted_image'])
            batch['wave_discr_output_real'] = F.interpolate(wave_discr_real_out, size=out_size, mode='nearest')
            batch['wave_discr_output_fake'] = F.interpolate(wave_discr_fake_out, size=out_size, mode='nearest')
            batch['wave_discr_output_diff'] = batch['wave_discr_output_real'] - batch['wave_discr_output_fake']

    def get_ddp_rank(self):
        return self.trainer.global_rank if (self.trainer.num_nodes * self.trainer.num_processes) > 1 else None

    def estimate_fisher(self):
        raise NotImplementedError()

    def consolidate(self, fisher):
        raise NotImplementedError()
    




