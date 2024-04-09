import logging
import random
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from .transformations import fetch_transform
import torch.nn.functional as F


_logger = logging.getLogger(__name__)


def upsample2d_flow_as(inputs, target, mode="bilinear", if_rate=False):
    h, w = target
    if if_rate:
        _, _, h_, w_ = inputs.size()
        inputs[:, 0, :, :] *= (w / w_)
        inputs[:, 1, :, :] *= (h / h_)
    res = F.interpolate(inputs, [h, w], mode=mode, align_corners=True)
    return res

def worker_init_fn(worker_id):
    rand_seed = random.randint(0, 2 ** 32 - 1)
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


class HomoValData(Dataset):
    def __init__(self, params, transform, phase):
        assert phase in ["test", "val"]
        self.transform = transform
        self.base_path = params.data_dir
        self.list_path = self.base_path + 'test_list.txt'
 
        self.mv_data = np.load(self.base_path + 'test_mv_b.npy', allow_pickle=True).item()

        self.data_infor = open(self.list_path, 'r').readlines()
        self.crop_size = params.crop_size
        self.mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
        self.std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)
        self.shift = params.shift

    def __len__(self):
        # return size of dataset
        return len(self.data_infor)

    def __getitem__(self, idx):
        img_names = self.data_infor[idx].replace('\n', '')

        video_names = img_names.split('_')[0]

        img1_name = img_names.split('/')[0]
        
        img2_name = img_names.split('/')[1]

        pt_names = img1_name + '_' + img2_name + '.npy'

        img1 = cv2.imread(self.base_path + 'img/' + img1_name)
        img2 = cv2.imread(self.base_path + 'img/' + img2_name)

        #mv
        mv_name = img1_name + "_" + img2_name + ".npy"
        mv_flow = self.mv_data[mv_name]
        mv_flow = np.repeat(mv_flow, 8, axis = 1)
        mv_flow = np.repeat(mv_flow, 8, axis = 2)
        mv_flow = torch.Tensor(mv_flow).float()
        mv_flow = mv_flow/4.0
        mv_flow_rs = mv_flow.clone()
        mv_flow_rs = upsample2d_flow_as(mv_flow_rs.unsqueeze(dim = 0), (self.crop_size[0], self.crop_size[1]), if_rate = True) 
        mv_flow_rs = mv_flow_rs.squeeze(dim = 0)
        #

        imgs_full = torch.cat((torch.Tensor(img1), torch.Tensor(img2)), dim=-1).permute(2, 0, 1).float()
        ori_h, ori_w, _ = img1.shape

        pt_set = np.load(self.base_path + 'Coordinate-v2/' + pt_names, allow_pickle=True)
        pt_set = str(pt_set.item())

        img1_rgb = cv2.resize(img1, (self.crop_size[1], self.crop_size[0]))
        img2_rgb = cv2.resize(img2, (self.crop_size[1], self.crop_size[0]))

        img1 = (img1 - self.mean_I) / self.std_I
        img2 = (img2 - self.mean_I) / self.std_I

        img1 = np.mean(img1, axis=2, keepdims=True)  
        img2 = np.mean(img2, axis=2, keepdims=True)

        img1_rs = cv2.resize(img1, (self.crop_size[1], self.crop_size[0]))
        img2_rs = cv2.resize(img2, (self.crop_size[1], self.crop_size[0]))

        img1, img2, img1_rs, img2_rs, img1_rgb, img2_rgb = list(
            map(torch.Tensor, [img1, img2, img1_rs, img2_rs, img1_rgb, img2_rgb]))

        imgs_gray_full = torch.cat((img1, img2), dim=-1).permute(2, 0, 1).float()
        imgs_gray_patch = torch.cat((img1_rs.unsqueeze(0), img2_rs.unsqueeze(0)), dim=0).float()
        imgs_patch_rgb = torch.cat([img1_rgb, img2_rgb], dim=-1).permute(2, 0, 1).float()

        ori_size = torch.Tensor([ori_w, ori_h]).float()
        Ph, Pw = img1_rs.size()

        pts = torch.Tensor([[0, 0], [Pw - 1, 0], [0, Ph - 1], [Pw - 1, Ph - 1]]).float()
        start = torch.Tensor([0, 0]).reshape(2, 1, 1).float()
        data_dict = {"imgs_patch_rgb": imgs_patch_rgb, "imgs_gray_full": imgs_gray_full, "imgs_full": imgs_full,
                     "imgs_gray_patch": imgs_gray_patch, "ori_size": ori_size,
                     "pt_set": pt_set, "video_names": video_names, 'pt_names': pt_names, "pts": pts, "start": start, 
                     "mv_flow": mv_flow, "mv_flow_patch": mv_flow_rs, 
                     }
        return data_dict


def fetch_dataloader(params):

    _logger.info("Dataset type: {}, transform type: {}".format(params.dataset_type, params.transform_type))
    train_transforms, test_transforms = fetch_transform(params)

    if params.dataset_type == "homo":
        val_ds = HomoValData(params, phase='val', transform=test_transforms)

    dataloaders = {}

    dl = DataLoader(
                    val_ds,
                    batch_size=params.eval_batch_size,
                    shuffle=False,
                    num_workers=params.num_workers,
                    pin_memory=params.cuda,
                    prefetch_factor=3,  # for pytorch >=1.5.0
                )
    
    dataloaders['val'] = dl

    return dataloaders
