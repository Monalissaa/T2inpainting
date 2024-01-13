import pandas as pd

file_path = '/mnt/d/post/codes/lama/outputs/lama-celebahq_full_config_CLEVR_data100_seed3_aug_fix_cl2l_cg2l_ud_wo_fm_stage_two_tsa_all/model0_random_thick_256_metrics.csv'
save_path = '/mnt/d/post/codes/lama/outputs/lama-celebahq_full_config_CLEVR_data100_seed3_aug_fix_cl2l_cg2l_ud_wo_fm_stage_two_tsa_all/model0_random_thick_256_metrics_different_mask_ratio.txt'
# 读取csv文件
df = pd.read_csv(file_path, sep='\t')

# 获取第二列的前5个值
values_fid = df.iloc[1:6, 1].values
values_lpips = df.iloc[1:6, 2].values

# 以'\t'作为间隔符输出
output_fid = '\t'.join(map(str, values_fid))
output_lpips = '\t'.join(map(str, values_lpips))

# 写入文件
with open(save_path, 'w') as f:
    f.write('fid:{}'.format(output_fid))
    f.write('\n')
    f.write('lpips:{}'.format(output_lpips))

# print(output)
