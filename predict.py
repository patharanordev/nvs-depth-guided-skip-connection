from PIL import Image
from torch.utils.data.dataloader import DataLoader
from options.test_options import TestOptions
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from util.visualizer import Visualizer
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF
from models.base_model import BaseModel
import time
from util import util
import os
import random
from scipy.spatial.transform import Rotation as ROT

class CustomDataset(Dataset):
    def __init__(self, data_arr, labels):
        self.data = data_arr
        self.labels = labels
        self.len = len(self.data)

    def __getitem__(self, index):
        return { 'data':self.data[index], 'target':self.labels[index] }

    def __len__(self):
        return len(self.data)


def inference(model, data):
    stime = time.perf_counter()
    fake_target_view_img = model.predict(data)
    etime = time.perf_counter()
    print('Inference time : {:.2f} sec.'.format((etime-stime)))
    img_path = os.path.join(img_dir, 'predict.png')
    util.save_image(fake_target_view_img, img_path)

def pred_from_dataset(opt):
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    with torch.no_grad():
        # Test some dataset
        for _, data in enumerate(tqdm(dataset)):
            # Inference
            inference(model, data)
            break

def pred_from_file(opt, model, img_fpath):
    pil_image = Image.open(img_fpath)
    img = np.asarray(pil_image.convert('RGB'))
    pil_image.close()

    # Force target view angle
    azim_b = 0
    azim_a = 140
    # azim_a = random.choice([
    #     0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200,
    #     220, 240, 260, 280, 300, 320, 340
    # ])
    elev_a = 20 if not opt.random_elevation else random.randint(0, 2)*10
    elev_b = 20 if not opt.random_elevation else random.randint(0, 2)*10

    A = img / 255. * 2 - 1
    A = torch.from_numpy(A.astype(np.float32)).permute((2, 0, 1))
    T = np.array([0, 0, 2]).reshape((3, 1))

    RA = ROT.from_euler('xyz', [-elev_a, azim_a, 0],degrees=True).as_dcm()
    RB = ROT.from_euler('xyz', [-elev_b, azim_b, 0],degrees=True).as_dcm()
    R = RA.T @ RB
    T = -R.dot(T) + T
    mat = np.block([[R, T], [np.zeros((1, 3)), 1]])
    data = {'A': A, 'RT': mat.astype(np.float32)}

    dataset = CustomDataset([data], ['chair'])
    loader = DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.nThreads)
    )

    with torch.no_grad():
        for batch in loader:
            print(batch['data'])
            # Inference
            inference(model, batch['data'])
            break


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

model = BaseModel(opt)

# # Predict image from dataset
# pred_from_dataset(opt)

# Predict image from file
pred_from_file(opt, model, './example/images/2aa122c56910bad8fb0eb7e753c06942_280_20.png')