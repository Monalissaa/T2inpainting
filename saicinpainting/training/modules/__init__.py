import logging

from saicinpainting.training.modules.ffc import FFCResNetGenerator, FFCResNetGeneratorSecondStage, FFCResNetGeneratorSpottune, FFCResNetGeneratorSimpleAdd, \
    FFCResNetGeneratorMaxChangeAdd, FFCResNetGeneratorSmall, FFCResNetGeneratorDropout
from saicinpainting.training.modules.pix2pixhd import GlobalGenerator, MultiDilatedGlobalGenerator, \
    NLayerDiscriminator, MultidilatedNLayerDiscriminator, WaveNLayerDiscriminator

def make_generator(config, kind, **kwargs):
    logging.info(f'Make generator {kind}')

    if kind == 'pix2pixhd_multidilated':
        return MultiDilatedGlobalGenerator(**kwargs)
    
    if kind == 'pix2pixhd_global':
        return GlobalGenerator(**kwargs)

    if kind == 'ffc_resnet':
        if config.new_params.two_stage:
            return FFCResNetGeneratorSecondStage(**kwargs)
        elif config.new_params.spottune:
            return FFCResNetGeneratorSpottune(**kwargs)
        elif config.new_params.simple_add:
            return FFCResNetGeneratorSimpleAdd(**kwargs)
        elif config.new_params.max_change_add:
            return FFCResNetGeneratorMaxChangeAdd(**kwargs)
        elif config.new_params.CWD>0:
            return FFCResNetGeneratorSmall(**kwargs)
        elif config.new_params.p_dropout>0:
            kwargs['p_dropout'] = config.new_params.p_dropout
            return FFCResNetGeneratorDropout(**kwargs)
        else:
            return FFCResNetGenerator(**kwargs)

    raise ValueError(f'Unknown generator kind {kind}')


def make_discriminator(kind, **kwargs):
    logging.info(f'Make discriminator {kind}')

    if kind == 'pix2pixhd_nlayer_multidilated':
        return MultidilatedNLayerDiscriminator(**kwargs)

    if kind == 'pix2pixhd_nlayer':
        return NLayerDiscriminator(**kwargs)
    
    if kind == 'wave':
        return WaveNLayerDiscriminator(**kwargs)
    

    raise ValueError(f'Unknown discriminator kind {kind}')
