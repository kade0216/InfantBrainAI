import nibabel as nib
import numpy as np

from config import *
import model_package.models.models as models
import model_package.data.data_util as data_util

class ScoringService(object):
    '''a singleton pattern class for prediction
    
    '''
    model = None

    @classmethod
    def get_model(cls):
        '''get or create a singleton instance of model
        
        '''
        if cls.model == None:

            G = models.create_model(opt=opt)
            G.to(device)
            # print(G)
            with open(os.path.join(model_path, "latest.pth"), "rb") as inp:
                G.load_state_dict(torch.load(inp, map_location=device))
                cls.model = G
            # print('model and weight connected')
            return cls.model

    @classmethod
    def predict(cls, input):
        '''make a single prediction
        
        '''
        G = cls.get_model()

        with torch.no_grad():
            print('model loaded')
            pred_name = '7Subcortical_Seg' + opt.extension
            try:
                nib.load(input)
            except ValueError:
                nib.Nifti1Header.quaternion_threshold = -1e-06
            tmp_scans = np.squeeze(nib.load(input).get_fdata())
            tmp_scans[tmp_scans < 0] = 0

            # define matrix to store prediction and normalization matrices
            pred = np.zeros((opt.cls_num,)+tmp_scans.shape)
            tmp_norm = np.zeros((opt.cls_num,) + tmp_scans.shape)
            # normalize image
            if opt.normalize:
                tmp_scans = data_util.norm_img(tmp_scans, opt.norm_perc)
            scan_patches, tmp_path, tmp_idx = data_util.patch_slicer(
                tmp_scans, tmp_scans, opt.patch_size, opt.patch_stride,
                remove_bg=True, test=True, ori_path=None
            )
            print('matrix defined')

            # go through all patches
            for idx, patch in enumerate(scan_patches):
                ipt = torch.from_numpy(patch).to(device=device, dtype=torch.float)
                tmp_pred = G(ipt.reshape((1,1,)+ipt.shape))
                patch_idx = tmp_idx[idx]
                patch_idx = (slice(0, opt.cls_num),) + (slice(patch_idx[0], patch_idx[1]), slice(patch_idx[2], patch_idx[3]), slice(patch_idx[4], patch_idx[5]))
                pred[patch_idx] += torch.squeeze(tmp_pred).detach().cpu().numpy()
                tmp_norm[patch_idx] += 1

            print('finished patch inferences')

            pred[tmp_norm > 0] = (pred[tmp_norm > 0]) / tmp_norm[tmp_norm > 0]
            sf = torch.nn.Softmax(dim=0)
            pred_vol = sf(torch.from_numpy(pred)).numpy()
            pred_vol = np.argmax(pred_vol, axis=0).astype(np.float)
            ori_scan = nib.load(input)
            sav_img = nib.Nifti1Image(np.round(pred_vol).astype(np.int), ori_scan.affine, header=ori_scan.header)
            nib.save(sav_img, os.path.join(out_path, pred_name))

            print('saved prediction output')
