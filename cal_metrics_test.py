import nibabel as nib
import numpy as np
import os
import pandas as pd


def cal_dice(pred, tar, tag):
    return (2 * np.sum(np.multiply(pred == tag, tar == tag), axis=None) + 1e-5) / (
                np.sum(pred == tag, axis=None) + np.sum(tar == tag, axis=None) + 1e-5)

ext = '.nii.gz'
gt_path = '/media/win_ssd/Ubuntu/Subcortical_Seg/50_T2_infants/remapped_label'
opt_path = '/media/win_ssd/Ubuntu/Subcortical_Seg/Longitudinal_scans/results'
tag_num = 7
sbjs = [i for i in os.listdir(gt_path) if i.endswith(ext)]
sbjs = [i.split(ext)[0] for i in sbjs]

sbjs = list(set(sbjs))

for i, sbj_id in enumerate(sbjs):
    print(sbj_id)
    GT_scan = np.round(nib.load(os.path.join(gt_path, sbj_id + ext)).get_fdata())
    Fake_scan = np.round(nib.load(os.path.join(opt_path, sbj_id + '_7Subcortical_Seg.nii.gz')).get_fdata())
    print(np.unique(GT_scan))
    DSC = []
    for j in range(1,tag_num+1):
        DSC.append(cal_dice(GT_scan, Fake_scan, j))
    d = {'sbj:': sbj_id}
    for j in range(tag_num):
        d.update({'DSC'+str(j+1): DSC[j]})

    d.update({'meanDSC': np.mean(DSC)})
    if i == 0:
        df = pd.DataFrame(data=d, index=[0])
    else:
        df2 = pd.DataFrame(data=d, index=[i])
        df = df.append(df2)

df.to_excel(os.path.join(opt_path, 'results.xlsx'))

