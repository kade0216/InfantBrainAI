import os
import torch
import model_package.options.test_options as options

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")
input_path = os.path.join(prefix, 'input')
out_path = os.path.join(prefix, 'output')

opt = options.TestOptions().parse(save=False)
opt.nThreads = 0  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')