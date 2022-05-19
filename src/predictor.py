# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function
import io
import uuid
import os
import zipfile
import nibabel as nib

import flask
from flask import request
import torch
import numpy as np

import model_package.models.models as models
import model_package.options.test_options as options
import model_package.data.data_util as data_util

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")
input_path = os.path.join(prefix, 'input')
out_path = os.path.join(prefix, 'output')

opt = options.TestOptions().parse(save=False)
opt.nThreads = 0  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ScoringService(object):
    model = None

    @classmethod
    def get_model(cls):
        '''get or create a singleton instance of model
        
        '''
        if cls.model == None:

            G = models.create_model(opt=opt)
            G.to(device)
            print(G)
            with open(os.path.join(model_path, "latest.pth"), "rb") as inp:
                G.load_state_dict(torch.load(inp, map_location=device))
                cls.model = G
            return cls.model

    @classmethod
    def predict(cls, input):
        '''make a single prediction
        
        '''
        G = cls.get_model()

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

        # go through all patches
        for idx, patch in enumerate(scan_patches):
            ipt = torch.from_numpy(patch).to(device=device, dtype=torch.float)
            tmp_pred = G(ipt.reshape((1,1,)+ipt.shape))
            patch_idx = tmp_idx[idx]
            patch_idx = (slice(0, opt.cls_num),) + (slice(patch_idx[0], patch_idx[1]), slice(patch_idx[2], patch_idx[3]), slice(patch_idx[4], patch_idx[5]))
            pred[patch_idx] += torch.squeeze(tmp_pred).detach().cpu().numpy()
            tmp_norm[patch_idx] += 1

        pred[tmp_norm > 0] = (pred[tmp_norm > 0]) / tmp_norm[tmp_norm > 0]
        sf = torch.nn.Softmax(dim=0)
        pred_vol = sf(torch.from_numpy(pred)).numpy()
        pred_vol = np.argmax(pred_vol, axis=0).astype(np.float)
        ori_scan = nib.load(input)
        sav_img = nib.Nifti1Image(np.round(pred_vol).astype(np.int), ori_scan.affine, header=ori_scan.header)
        nib.save(sav_img, os.path.join(out_path, pred_name))

app = flask.Flask(__name__)
 
@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully.
    
    """
    if request.method == 'GET':
        health = ScoringService.get_model() is not None

        status = 200 if health else 404
        return flask.Response(response="pong\n", status=status, mimetype="application/json")
    else:
        return flask.Response(response='method not allowed\n', status=405, mimetype="application/json")

@app.route("/invocations", methods=["POST"])
def transformation():
    '''make predictions
    
    '''
    if request.method == 'POST':
        if flask.request.content_type == "application/zip":
            if not os.path.isdir(input_path):
                os.makedirs(input_path)

            bytes = flask.request.data
            zippedData = zipfile.ZipFile(io.BytesIO(bytes), 'r')
            
            niiFilePaths = []
            for zippedFileName in zippedData.namelist():
                content = zippedData.open(zippedFileName).read()
                niiFilePath = '{}.nii'.format(os.path.join(input_path, str(uuid.uuid4())))
                niiFilePaths.append(niiFilePath)
                fileOut = open(niiFilePath, 'wb')
                fileOut.write(content)
                fileOut.close()

            for niiFilePath in niiFilePaths:
                ScoringService.predict(niiFilePath)

            return flask.Response(response="prediction generated\n", status=200, mimetype="application/json")
        else:
            return flask.Response(response="unsupported media type\n", status=415, mimetype="application/json")
    else:
        return flask.Response(response='method not allowed\n', status=405, mimetype="application/json")
