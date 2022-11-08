import logging

from saicinpainting.training.modules.ffc import FFCResNetGenerator, FFCResNetGeneratorSecondStage, FFCResNetGeneratorSpottune, FFCResNetGeneratorSimpleAdd, \
    FFCResNetGeneratorMaxChangeAdd, FFCResNetGeneratorSmall, FFCResNetGeneratorDropout, FFCResNetFSMRGenerator, FFCResNetFixAddFFCGenerator, \
        TsaFourFFCResNetGenerator, TsaTwoFFCResNetGenerator, TsaOneFFCResNetGenerator, TsaG2GFFCResNetGenerator, TsaG2GL2GFFCResNetGenerator, \
            TsaG2GL2LFFCResNetGenerator, TsaG2GConvL2LFFCResNetGenerator, TsaG2GDownUpFFCResNetGenerator, TsaG2GConvL2GFFCResNetGenerator, \
                TsaMiddleAllConvFFCResNetGenerator, TsaAllConvFFCResNetGenerator, TsaAllGroupConvFFCResNetGenerator, TsaAllConvL2LG2LFFCResNetGenerator
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
        elif config.new_params.fsmr.pl>0:
            return FFCResNetFSMRGenerator(**kwargs)
        elif config.new_params.fix_add_ffc:
            return FFCResNetFixAddFFCGenerator(**kwargs)
        elif config.new_params.tsa.four:
            return TsaFourFFCResNetGenerator(**kwargs)
        elif config.new_params.tsa.two:
            return TsaTwoFFCResNetGenerator(**kwargs)
        elif config.new_params.tsa.one:
            return TsaOneFFCResNetGenerator(**kwargs) 
        elif config.new_params.tsa.l2g:
            return TsaG2GL2GFFCResNetGenerator(**kwargs)
        elif config.new_params.tsa.l2l:
            return TsaG2GL2LFFCResNetGenerator(**kwargs)
        elif config.new_params.tsa.convl2l:
            return TsaG2GConvL2LFFCResNetGenerator(**kwargs)
        elif config.new_params.tsa.g2g_down_up:
            return TsaG2GDownUpFFCResNetGenerator(**kwargs)
        elif config.new_params.tsa.convl2g:
            return TsaG2GConvL2GFFCResNetGenerator(**kwargs)
        elif config.new_params.tsa.middle_all:
            return TsaMiddleAllConvFFCResNetGenerator(**kwargs)
        elif config.new_params.tsa.all:
            return TsaAllConvFFCResNetGenerator(**kwargs)
        elif config.new_params.tsa.all_group:
            kwargs['group_size'] = config.new_params.tsa.group_size
            return TsaAllGroupConvFFCResNetGenerator(**kwargs)
        elif config.new_params.tsa.release_tsa_global_bn:
            return TsaAllConvL2LG2LFFCResNetGenerator(**kwargs)
        elif config.new_params.tsa.g2g:
            return TsaG2GFFCResNetGenerator(**kwargs) 
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
