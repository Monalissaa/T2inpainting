import logging
from numpy import rate
import torch
from tqdm import trange
from saicinpainting.training.trainers.default import DefaultInpaintingTrainingModule


def get_training_model_class(kind):
    if kind == 'default':
        return DefaultInpaintingTrainingModule

    raise ValueError(f'Unknown trainer module {kind}')


def make_training_model(config):
    kind = config.training_model.kind
    kwargs = dict(config.training_model)
    kwargs.pop('kind')
    kwargs['use_ddp'] = config.trainer.kwargs.get('accelerator', None) == 'ddp'

    logging.info(f'Make training model {kind}')

    cls = get_training_model_class(kind)
    return cls(config, **kwargs)




import re
from numpy import mean
from matplotlib import pyplot as plt 

def load_checkpoint(train_config, path, map_location='cuda', strict=True):
    model: torch.nn.Module = make_training_model(train_config)
    state = torch.load(path, map_location=map_location)
    # model.load_state_dict(state['state_dict'], strict=strict)
    # model.on_load_checkpoint(state)
    # model.estimate_fisher()
    # path_adapt = '/home/mona/codes/lama/experiments/big-lama-transfer/models/last.ckpt'
    # path_adapt = '/mnt/d/post/codes/lama/experiment/lama-transfer-aug_fix_middleBlocks_convl2l_convg2l_fix_UpDown_without_featureMatching_loss_stage_two_aug_no_fix_without_featureMatching_loss/04-57-24/models/last.ckpt'
    # state_adapt = torch.load(path_adapt, map_location=map_location)['state_dict']

    # state_change = {}

    # for k, v in state['state_dict'].items():
    #     if ('generator.model' in k and '.weight' in k and 'bn' not in k and v.dim()!=1):
    #         # print(k, ' ', v.shape, '   adapt:', state_adapt[k].shape)

    #         rate_change = (abs(v-state_adapt[k])/abs(v)).mean().item()  # to float

    #         pattern = r'\.[0-9]*\.'
    #         layer_num = re.search(pattern, k).group()[1:-1]
    #         if state_change.get(layer_num)==None:
    #             state_change[layer_num] = [rate_change]
    #         else:
    #             state_change[layer_num].append(rate_change)
    #         # print(state_change[layer_num])
    
    # xx = []
    # yy = []

    # for k, v in state_change.items():
    #     # print(type(k), type(v[0]))
    #     xx.append('{}'.format(k))
    #     yy.append(mean(v))
    #     print('rate of changes on weights block {}'.format(k) + ': ' + str(mean(v)))
    
    # plt.bar(xx, yy, align =  'center') 
    # plt.title('rate of changes on weights(%)')
    # # plt.ylabel('Fisher information') 
    # # plt.show()
    # plt.savefig('/home/mona/codes/lama/rate_of_changes_on_weights_our2.png')
    # # plt.xlabel('X axis') 
    # exit(0)


    if strict == True: # training state
        if train_config.new_params.spottune or train_config.new_params.simple_add:
            model.load_state_dict(state['state_dict'], strict=False)
            for params, freeze_params in zip(model.generator.model[5:-13].parameters(), model.generator.pretrained_blocks.parameters()):
                freeze_params.data.copy_(params.data)  # initialize
                # print(self_param==transfer_param)
                freeze_params.requires_grad = False  # not update by gradient
        elif train_config.new_params.max_change_add:
            model.load_state_dict(state['state_dict'], strict=False)

            for params, freeze_params in zip(model.generator.model[5].parameters(), model.generator.max_change_blocks[0].parameters()):
                freeze_params.data.copy_(params.data)  # initialize
                freeze_params.requires_grad = False  # not update by gradient

            for params, freeze_params in zip(model.generator.model[20].parameters(), model.generator.max_change_blocks[1].parameters()):
                freeze_params.data.copy_(params.data)  # initialize
                # print(self_param==transfer_param)
                freeze_params.requires_grad = False  # not update by gradient

            for params, freeze_params in zip(model.generator.model[22].parameters(), model.generator.max_change_blocks[2].parameters()):
                freeze_params.data.copy_(params.data)  # initialize
                # print(self_param==transfer_param)
                freeze_params.requires_grad = False  # not update by gradient
        elif train_config.new_params.wave_dis:
            model.load_state_dict(state['state_dict'], strict=False)
            params_wave = []
            for name, params in model.wave_discriminator.named_parameters():
                if 'wave' not in name:
                    params_wave.append(params)

            for params, wave_params in zip(model.discriminator.parameters(), params_wave):
                wave_params.data.copy_(params.data)  # initialize
        elif train_config.new_params.contrast:
            model.load_state_dict(state['state_dict'], strict=False)
            # for params, contrast_params in zip(model.discriminator.parameters(), model.contrast_discriminator.parameters()):
            #     contrast_params.data.copy_(params.data)  # initialize
        elif train_config.new_params.randomInit:
            if train_config.new_params.randomInitUniform:
                for m in model.modules():
                    if isinstance(m, torch.nn.Conv2d):
                        torch.nn.init.kaiming_uniform_(m.weight, mode = 'fan_in',nonlinearity='relu')

            stete_need_transfer = {}
            for k, v in state['state_dict'].items():
                if ('generator.model.' in k): # skip up & down blocks
                    pattern = r'\.[0-9]*\.'
                    layer_num = re.search(pattern, k).group()[1:-1]
                    if train_config.new_params.randomInitOnlyUp:
                        if int(layer_num)>22:
                            continue
                    elif train_config.new_params.randomInitDown:
                        if int(layer_num)<5:
                            continue
                    elif train_config.new_params.randomStartEnd[0]!=0:
                        start = train_config.new_params.randomStartEnd[0]
                        end = train_config.new_params.randomStartEnd[1]
                        if int(layer_num)>=start and int(layer_num)<=end:
                            # print(layer_num)
                            continue
                    else:
                        if int(layer_num)<5 or int(layer_num)>22:
                            continue
                    
                stete_need_transfer[k] = v
            model.load_state_dict(stete_need_transfer, strict=False)
        elif train_config.new_params.ewc_onlyUp or train_config.new_params.celeba or train_config.new_params.p_dropout>0:
            model.load_state_dict(state['state_dict'], strict=False)
            # model.on_load_checkpoint(state)
        
        elif train_config.new_params.two_stage_from_init:
            model.load_state_dict(state['state_dict'], strict=strict)
        elif train_config.new_params.CWD:
            
            model.load_state_dict(state['state_dict'], strict=False)
            
            for params, freeze_params in zip(model.generator.small_model[:5].parameters(), model.generator.model[:5].parameters()):
                params.data.copy_(freeze_params.data)  # initialize
                # print(self_param==transfer_param)

            for params, freeze_params in zip(model.generator.small_model[-14:].parameters(), model.generator.model[-14:].parameters()):
                params.data.copy_(freeze_params.data)  # initialize
                # print(self_param==transfer_param)


            state_transferred = torch.load(train_config.new_params.resume_from_transferred, map_location=map_location)
            model.load_state_dict(state_transferred['state_dict'], strict=False)
        elif train_config.new_params.from_cwd:
            # model.load_state_dict(state['state_dict'], strict=False)
            stete_need_transfer = {}
            for k, v in state['state_dict'].items():
                
                if ('generator.small_model.' in k): # skip up & down blocks
                    key_param = k
                    key_param = key_param[:10] + key_param[16:]
                    # print(key_param)
                    # exit(0)
                    stete_need_transfer[key_param] = v
                elif ('generator.model.' in k):
                    continue
                else:
                    stete_need_transfer[k] = v
            model.load_state_dict(stete_need_transfer, strict=True)
                
        else:
            model.load_state_dict(state['state_dict'], strict=strict)
            model.on_load_checkpoint(state)
    else:
        model.load_state_dict(state['state_dict'], strict=strict)
        model.on_load_checkpoint(state)

    # if strict == True: # training state
    #     stete_need_transfer = {}
    #     for k, v in state['state_dict'].items():
    #         if ('generator.model.' in k):
    #             stete_need_transfer[k] = v
    #         # print(k)
    #     model.load_state_dict(stete_need_transfer, strict=False)
    # else: # predict state
    #     model.load_state_dict(state['state_dict'], strict=strict)
    #     model.on_load_checkpoint(state)
    return model
