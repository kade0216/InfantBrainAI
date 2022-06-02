# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function
import io
import uuid
import os
import zipfile

import flask
from flask import request

from scoring_service import ScoringService
from config import *
import worker

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
                if debug:
                    worker.make_prediction.delay(niiFilePath)
                else:
                    ScoringService.predict(niiFilePath)

            return flask.Response(response="prediction generated\n", status=200, mimetype="application/json")
        else:
            return flask.Response(response="unsupported media type\n", status=415, mimetype="application/json")
    else:
        return flask.Response(response='method not allowed\n', status=405, mimetype="application/json")
