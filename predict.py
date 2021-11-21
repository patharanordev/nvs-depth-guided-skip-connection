from options.test_options import TestOptions
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from util.visualizer import Visualizer
from tqdm import tqdm
import numpy as np
import torch
from models.base_model import BaseModel
import time
from util import util
import os

def inference(model, data):
    # stime = time.perf_counter()
    # model.forward()
    # etime = time.perf_counter()
    # print('Inference time : {:.2f} sec.'.format((etime-stime)))
    # epoch = opt.which_epoch
    # visuals = model.get_current_visuals()
    # for label, image_numpy in visuals.items():
    #     img_path = os.path.join(img_dir, 'epoch%s_%s.png' % (epoch, label))
    #     util.save_image(image_numpy, img_path)

    stime = time.perf_counter()
    fake_target_view_img = model.predict(data)
    etime = time.perf_counter()
    print('Inference time : {:.2f} sec.'.format((etime-stime)))
    img_path = os.path.join(img_dir, 'predict.png')
    util.save_image(fake_target_view_img, img_path)



opt = TestOptions().parse()
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True
opt.isTrain = False
opt.max_dataset_size = float("inf")

pred_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
img_dir = os.path.join(pred_dir, 'images')
anim_dir = os.path.join(pred_dir, 'anim')
print('create predict result directory %s...' % pred_dir)
util.mkdirs([pred_dir, img_dir, anim_dir])

data_loader = CustomDatasetDataLoader(opt)
dataset = data_loader.load_data()

model = BaseModel(opt)

L1s = []
SSIMs = []
with torch.no_grad():

    # Test some dataset
    for idx, data in enumerate(tqdm(dataset)):
        ida = data['id_a'][0].split('_')
        idb = data['id_b'][0].split('_')

        assert (ida[0] == idb[0])
        model_id = ida[0]
        ida = '_'.join(ida[1:])
        idb = '_'.join(idb[1:])

        model.set_input(data)

        # Inference
        inference(model)

        model.switch_mode('eval')

        model.anim_dict = {'vis': []}
        model.real_A = model.real_A[:1]
        model.real_B = model.real_B[:1]

        eval_res = model.evaluate()
        L1s.append(eval_res['L1'])
        SSIMs.append(eval_res['SSIM'])

        break

print('L1:{l1}, SSIM:{ssim}'.format(l1=np.mean(L1s), ssim=np.mean(SSIMs)))

