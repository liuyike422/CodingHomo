"""Evaluates the model"""

import argparse
import logging
import os

import cv2
import imageio
import torch.optim as optim
import itertools
import torch
from torch import nn
from model.utils import get_warp_flow 
from tqdm import tqdm
import numpy as np
import dataset.data_loader_homoGAN as data_loader
import model.net_mv as net
from common import utils
from loss.losses import compute_eval_results, compute_eval_results_bi, compute_eval_results_homoGAN
from common.manager import Manager
import flow_viz
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image
import imageio

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/pre_train/', help="Directory containing params.json")
parser.add_argument('--restore_file',
                    # default='experiments/pre_train/pre_train_0.5017.pth',
                    default='experiments/pre_train/result_0.3054.pth',
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'experiments/version1/model_latest.pth'
parser.add_argument('-ow', '--only_weights', action='store_true', default=True, help='Only use weights to load or load all train status.')


def make_gif(img1, img2, name):
    
    with imageio.get_writer(name+'.gif', mode='I', duration = 1,loop = 0) as writer:    
    
        writer.append_data(img1.astype(np.uint8))    
        writer.append_data(img2.astype(np.uint8))

def viz(flo):
    flo = flo.permute(1,2,0).cpu().numpy()    
    flo = flow_viz.flow_to_image(flo)
    return flo

def torch_warp(x, flo):
        x = torch.tensor(x).float()
        flo = torch.tensor(flo).cpu()
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.shape 

        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        #if x.is_cuda:
        #    grid = grid.cuda()

        vgrid = grid + flo

        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
        output = nn.functional.grid_sample(x, vgrid, padding_mode='zeros',align_corners=True)
        return output


# def make_align_heatmap(ref, inp):
#     heat_map = inp.astype(np.float32) - ref.astype(np.float32)
#     heat_map = abs(heat_map)
#     heat_map -= heat_map.min()
#     heat_map /= heat_map.max()
#     heat_map = np.clip(heat_map, 0.1, 1)
#     heat_map = cv2.applyColorMap((heat_map * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1].astype(np.uint8)
#     # heat_map = append_text_information(heat_map, text, ssim_scores)
#     return heat_map


def make_align_heatmap(ref, inp):
    heat_map = inp.astype(np.float32) - ref.astype(np.float32)
    heat_map = abs(heat_map)
    heat_map -= heat_map.min()
    heat_map /= heat_map.max()
    heat_map = np.clip(heat_map, 0.01, 1)
    # tmp = heat_map.copy()
    # heat_map[:,:,0] = tmp[:,:,2]
    # heat_map[:,:,2] = tmp[:,:,0]
    heat_map = cv2.applyColorMap((heat_map * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1].astype(np.uint8)
    # heat_map = append_text_information(heat_map, text, ssim_scores)
    tmp = heat_map.copy()
    heat_map[:,:,0] = tmp[:,:,2]
    heat_map[:,:,2] = tmp[:,:,0]

    return heat_map


def evaluate(model, manager):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        manager: a class instance that contains objects related to train and evaluate.
    """
    # set model to evaluation mode
    if manager.accelerator.is_main_process:
        manager.accelerator.print("\n")
        manager.logger.info("eval begin!")

    RE = ['0000011', '0000016', '00000147', '00000155', '00000158', '00000107', '00000239', '0000030']
    LT = ['0000038', '0000044', '0000046', '0000047', '00000238', '00000177', '00000188', '00000181']
    LL = ['0000085', '00000100', '0000091', '0000092', '00000216', '00000226']
    SF = ['00000244', '00000251', '0000026', '0000030', '0000034', '00000115']
    LF = ['00000104', '0000031', '0000035', '00000129', '00000141', '00000200']

    MSE_RE = []
    MSE_LT = []
    MSE_LL = []
    MSE_SF = []
    MSE_LF = []

    STD_RE = []
    STD_LT = []
    STD_LL = []
    STD_SF = []
    STD_LF = []
    STD_AVG = []


    torch.cuda.empty_cache()
    model.eval()

    step = 0

    idx = 0
    with torch.no_grad():
        # compute metrics over the dataset
        if manager.dataloaders["val"] is not None:
            # loss status and val status initial
            manager.reset_loss_status()
            manager.reset_metric_status("val")

            for data_batch in manager.dataloaders["val"]:
                # compute model output
                output = model(data_batch, step)

                step += 1

                video_name = data_batch["video_names"]
                # compute all loss on this batch
                eval_results = compute_eval_results_bi(data_batch, output)
                err_avg = eval_results["errors_m"]

                for j in range(len(err_avg)):
                    STD_AVG.append(err_avg[j].cpu().numpy())
                    if video_name[j] in RE:
                        MSE_RE.append(err_avg[j])
                        STD_RE.append(err_avg[j].cpu().numpy())
                    elif video_name[j] in LT:
                        MSE_LT.append(err_avg[j])
                        STD_LT.append(err_avg[j].cpu().numpy())
                    elif video_name[j] in LL:
                        MSE_LL.append(err_avg[j])
                        STD_LL.append(err_avg[j].cpu().numpy())
                    elif video_name[j] in SF:
                        MSE_SF.append(err_avg[j])
                        STD_SF.append(err_avg[j].cpu().numpy())
                    elif video_name[j] in LF:
                        MSE_LF.append(err_avg[j])
                        STD_LF.append(err_avg[j].cpu().numpy())

            MSE_RE_avg = sum(MSE_RE) / len(MSE_RE)
            MSE_LT_avg = sum(MSE_LT) / len(MSE_LT)
            MSE_LL_avg = sum(MSE_LL) / len(MSE_LL)
            MSE_SF_avg = sum(MSE_SF) / len(MSE_SF)
            MSE_LF_avg = sum(MSE_LF) / len(MSE_LF)

            # results shoud be gatherd from multiple GPUs to single GPU
            MSE_RE_avg = torch.mean(manager.accelerator.gather_for_metrics(
                torch.tensor(MSE_RE_avg).cuda(manager.accelerator.process_index)))
            MSE_LT_avg = torch.mean(manager.accelerator.gather_for_metrics(
                torch.tensor(MSE_LT_avg).cuda(manager.accelerator.process_index)))
            MSE_LL_avg = torch.mean(manager.accelerator.gather_for_metrics(
                torch.tensor(MSE_LL_avg).cuda(manager.accelerator.process_index)))
            MSE_SF_avg = torch.mean(manager.accelerator.gather_for_metrics(
                torch.tensor(MSE_SF_avg).cuda(manager.accelerator.process_index)))
            MSE_LF_avg = torch.mean(manager.accelerator.gather_for_metrics(
                torch.tensor(MSE_LF_avg).cuda(manager.accelerator.process_index)))

        if manager.accelerator.is_main_process:
            MSE_avg = (MSE_RE_avg + MSE_LT_avg + MSE_LL_avg + MSE_SF_avg + MSE_LF_avg) / 5
            # manager.accelerator.print(f"MSE_avg: {MSE_avg}")

            Metric = {
                "MSE_RE_avg": MSE_RE_avg,
                "MSE_LT_avg": MSE_LT_avg,
                "MSE_LL_avg": MSE_LL_avg,
                "MSE_SF_avg": MSE_SF_avg,
                "MSE_LF_avg": MSE_LF_avg,
                "AVG": MSE_avg
            }
            manager.update_metric_status(metrics=Metric, split="val")

            manager.logger.info("Loss/valid epoch_val {}: {:.4f}. RE:{:.4f} LT:{:.4f} LL:{:.4f} SF:{:.4f} LF:{:.4f} ".format(
                manager.epoch_val, MSE_avg, MSE_RE_avg, MSE_LT_avg, MSE_LL_avg, MSE_SF_avg, MSE_LF_avg))

            manager.print_metrics("val", title="val", color="green")

            manager.epoch_val += 1

            torch.cuda.empty_cache()
            torch.set_grad_enabled(True)
            model.train()
            val_metrics = {'MSE_avg': MSE_avg}

            STD_AVG_avg = np.std(STD_AVG)
            STD_RE_avg = np.std(STD_RE)
            STD_LL_avg = np.std(STD_LL)
            STD_LT_avg = np.std(STD_LT)
            STD_SF_avg = np.std(STD_SF)
            STD_LF_avg = np.std(STD_LF)


            Metric_std = {
                "STD_AVG": STD_AVG_avg,
                "STD_RE": STD_RE_avg,
                "STD_LL": STD_LL_avg,
                "STD_LT": STD_LT_avg,
                "STD_SF": STD_SF_avg,
                "STD_LF": STD_LF_avg
            }

            print(Metric_std)

            return val_metrics


def test(model, manager):
    """Test the model with loading checkpoints.

    Args:
        model: (torch.nn.Module) the neural network
        manager: a class instance that contains objects related to train and evaluate.
    """
    # set model to evaluation mode

    RE = ['0000011', '0000016', '00000147', '00000155', '00000158', '00000107', '00000239', '0000030']
    LT = ['0000038', '0000044', '0000046', '0000047', '00000238', '00000177', '00000188', '00000181']
    LL = ['0000085', '00000100', '0000091', '0000092', '00000216', '00000226']
    SF = ['00000244', '00000251', '0000026', '0000030', '0000034', '00000115']
    LF = ['00000104', '0000031', '0000035', '00000129', '00000141', '00000200']
    MSE_RE = []
    MSE_LT = []
    MSE_LL = []
    MSE_SF = []
    MSE_LF = []

    torch.cuda.empty_cache()
    model.eval()
    k = 0
    flag = 0

    rubost_tot = torch.zeros((9))
    for m in range(9):
        rubost_tot[m] = 0

    with torch.no_grad():
        # compute metrics over the dataset
        if manager.dataloaders["val"] is not None:
            # loss status and val status initial
            manager.reset_loss_status()
            manager.reset_metric_status("val")
            with tqdm(total=len(manager.dataloaders['val']), ncols=100) as t:
                for data_batch in manager.dataloaders["val"]:

                    video_name = data_batch["video_names"]
                    imgs_full = data_batch["imgs_full"]

                    b, c, h, w = imgs_full.shape

                    data_batch = utils.tensor_gpu(data_batch)
                    output_batch = model(data_batch, 0)

                    # Homo_b = output_batch["Homo_b"]

                    flag += 1
                    t.update()
                    eval_results = compute_eval_results_bi(data_batch, output_batch)
                    err_avg = eval_results["errors_m"]

                    # rubost_bs = eval_results["rubost"]
                    # for m in range(9):
                    #     rubost_tot[m] += rubost_bs[m]

                    for j in range(len(err_avg)):

                        k += 1
                        if video_name[j] in RE:
                            MSE_RE.append(err_avg[j])
                        elif video_name[j] in LT:
                            MSE_LT.append(err_avg[j])
                        elif video_name[j] in LL:
                            MSE_LL.append(err_avg[j])
                        elif video_name[j] in SF:
                            MSE_SF.append(err_avg[j])
                        elif video_name[j] in LF:
                            MSE_LF.append(err_avg[j])
                    
                    #visual mv
                    # for i in range(b):
                        
                    #     #load form output
                    #     tensorA, tensorB = data_batch["imgs_gray_full"][i, :1, ...], data_batch["imgs_gray_full"][i, 1:, ...]
                    #     feaA, feaB = output_batch["img1_full_fea"][i,:,:,:], output_batch["img2_full_fea"][i,:,:,:]
                    #     tensorArgb, tensorBrgb = data_batch['imgs_full'][i, :3, ...], data_batch['imgs_full'][i, 3:, ...]
                    #     homo_flow = output_batch["flow_b"][i,:,:,:].permute(2,0,1)
                    #     _, h, w = tensorA.shape
                    #     mv_flow = data_batch["mv_flow"][i,:,:,:]
                    #     hg_flow_b = data_batch['homo_flow_b'][i,:,:,:]
                    #     pts = data_batch["pt_set"]
                    #     # resi = abs(data_batch["resi"][i,:,:,:])#如果残差太小可以考虑正则化放大视觉
                    #     pt_name = data_batch["pt_names"][i]
                    #     error = err_avg[i]
                    #     imgBrgb = tensorBrgb.cpu().numpy().transpose(1,2,0)

                    #     imgA = tensorArgb.unsqueeze(dim = 0)
                    #     mv_b = get_warp_flow(imgA, homo_flow.unsqueeze(dim = 0)).squeeze(dim = 0).permute(1, 2, 0).cpu().numpy()

                    #     #make gif
                    #     imgB = imgBrgb.copy()
                    #     imgB[:,:,0] = imgBrgb[:,:,2]
                    #     imgB[:,:,2] = imgBrgb[:,:,0]
                    #     warped = mv_b.copy()
                    #     warped[:,:,0] = mv_b[:,:,2]
                    #     warped[:,:,2] = mv_b[:,:,0]

                    #     gif1 = np.concatenate([imgB], 1)#gt warp_back warp
                    #     gif2 = np.concatenate([warped], 1)
                    #     gif_name_cor = './test_gifs/' + f'{pt_name}' + "%.4f"% (error)
                    #     make_gif(gif1, gif2, gif_name_cor)
                    
                    # visual pip
                    for i in range(b):
                        
                        #load form output
                        tensorA, tensorB = data_batch["imgs_gray_full"][i, :1, ...], data_batch["imgs_gray_full"][i, 1:, ...]
                        feaA, feaB = output_batch["img1_full_fea"][i,:,:,:], output_batch["img2_full_fea"][i,:,:,:]
                        tensorArgb, tensorBrgb = data_batch['imgs_full'][i, :3, ...], data_batch['imgs_full'][i, 3:, ...]
                        homo_flow = output_batch["flow_b"][i,:,:,:].permute(2,0,1)
                        homo_flow_f = output_batch["flow_f"][i,:,:,:].permute(2,0,1)
                        _, h, w = tensorArgb.shape
                        mv_flow = data_batch["mv_flow"][i,:,:,:]
                        pts = data_batch["pt_set"]
                        pt_name = data_batch["pt_names"][i]
                        error = err_avg[i]

                        feaA = feaA/(feaA.max())
                        feaB = feaB/(feaB.max())

                        mv_flo_viz = viz(mv_flow)
                        homo_flo_viz = viz(homo_flow)

                    #     mask_b0 = output_batch['mask_b'][0][i,:,:,:].repeat(3,1,1).permute(1, 2, 0).cpu().numpy()
                    #     mask_b1 = output_batch['mask_b'][1][i,:,:,:].repeat(3,1,1).permute(1, 2, 0).cpu().numpy()
                        mask_b2 = output_batch['mask_b'][2][i,:,:,:].repeat(3,1,1).permute(1, 2, 0).cpu().numpy()
                        mask_b3 = output_batch['mask_b'][3][i,:,:,:].repeat(3,1,1).permute(1, 2, 0).cpu().numpy()
                        mask_b4 = output_batch['mask_b'][4][i,:,:,:].repeat(3,1,1).permute(1, 2, 0).cpu().numpy()
                        
                        #warp by flow
                        warpA = get_warp_flow(feaA.unsqueeze(dim = 0), homo_flow.unsqueeze(dim = 0)).squeeze(dim = 0).permute(1, 2, 0).cpu().numpy()
                        
                        imgArgb = tensorArgb.cpu().numpy().transpose(1,2,0)
                        imgBrgb = tensorBrgb.cpu().numpy().transpose(1,2,0)
                        feaA = feaA.repeat(3,1,1).permute(1, 2, 0).cpu().numpy()
                        feaB = feaB.repeat(3,1,1).permute(1, 2, 0).cpu().numpy()
                        
                        #save_img
                        outname = './result/' + f'{pt_name}' + "%.4f"% (error) + '_'
                        # cv2.imwrite(outname + 'pic1.png', imgArgb)
                        # cv2.imwrite(outname + 'pic2.png', imgBrgb)
                        cv2.imwrite(outname + 'warp1fea.png', warpA*255)
                        # make_gif(imgBrgb, warpA, outname)
                        # cv2.imwrite(outname + 'fea1.png', feaA*255)
                        # cv2.imwrite(outname + 'fea2.png', feaB*255)
                        # cv2.imwrite(outname + 'homo.png', homo_flo_viz)
                        # cv2.imwrite(outname + 'mv.png', mv_flo_viz)
                        outname = './mask/' + f'{pt_name}' + '_'
                        # cv2.imwrite(outname + 'mask0.png', mask_b0*255)
                    #     # cv2.imwrite(outname + 'mask1.png', mask_b1*255)
                        cv2.imwrite(outname + 'mask_m.png', mask_b2*255)
                        cv2.imwrite(outname + 'mask_c.png', mask_b3*255)
                        cv2.imwrite(outname + 'mask_cmc.png', mask_b4*255)    

                        outmaskname = './mask/' + f'{pt_name}' + '_'
                        mask = np.concatenate([mask_b2*255, mask_b3*255], 1)#gt warp_back warp         
                        cv2.imwrite(outmaskname + 'mask.png', mask_b3*350)               

                    #rgb warp pic
                    # for i in range(b):
                        
                    # #     #load form output
                    #     # tensorA, tensorB = data_batch["imgs_gray_full"][i, :1, ...], data_batch["imgs_gray_full"][i, 1:, ...]
                    #     tensorArgb, tensorBrgb = data_batch['imgs_full'][i, :3, ...], data_batch['imgs_full'][i, 3:, ...]
                    #     homo_flow_b = output_batch["flow_b"][i,:,:,:].permute(2,0,1)
                    #     homo_flow_f = output_batch["flow_f"][i,:,:,:].permute(2,0,1)
                    #     hg_flow_b = data_batch['homo_flow_b'][i,:,:,:]
                    #     hg_flow_f = data_batch['homo_flow_f'][i,:,:,:]
                    #     _, h, w = tensorArgb.shape
                    #     mv_flow = data_batch["mv_flow"][i,:,:,:]
                    #     pts = data_batch["pt_set"]
                    #     pt_name = data_batch["pt_names"][i]
                    #     error = err_avg[i]
                    #     error_hg = eval_results["errors_hg"][i]
                    #     dist = error - error_hg
                        # if dist > 0:
                        #     continue
                        # imgBrgb = tensorBrgb.cpu().numpy().transpose(1,2,0)
                        
                        # # #warp by flow
                        # imgA = tensorArgb.unsqueeze(dim = 0)
                        # # mvHomo_b = get_warp_flow(imgA, homo_flow_b.unsqueeze(dim = 0)).squeeze(dim = 0).permute(1, 2, 0).cpu().numpy()
                        # # hg_b = get_warp_flow(imgA, hg_flow_b.unsqueeze(dim = 0)).squeeze(dim = 0).permute(1, 2, 0).cpu().numpy()
                        # mv_b = get_warp_flow(imgA, mv_flow.unsqueeze(dim = 0)).squeeze(dim = 0).permute(1, 2, 0).cpu().numpy()

                        # # mvHomo_heat = make_align_heatmap(mvHomo_b, imgBrgb)
                        # # hg_heat = make_align_heatmap(hg_b, imgBrgb)
                        # mv_heat = make_align_heatmap(mv_b, imgBrgb)

                        # # mvHomo_b[:,:,2] = imgBrgb[:,:,2]
                        # # hg_b[:,:,2] = imgBrgb[:,:,2]
                        # mv_b[:,:,2] = imgBrgb[:,:,2]
                        # tot = np.concatenate([mvHomo_b, hg_b, mvHomo_heat, hg_heat], 1)

                        # outname = './sub_b/' + "%.4f"% (dist) + '_'  + f'{pt_name}' + "%.4f"% (error) + '_'
                        # cv2.imwrite('./red/' + f'{pt_name}' + '.png', mvHomo_b)
                        # # cv2.imwrite(outname + 'HomoGAN.png', hg_f)
                        # cv2.imwrite('./heat/' + f'{pt_name}' + '.png', mvHomo_heat)
                        # cv2.imwrite(outname + 'HomoGAN_heat.png', hg_heat)
                        # # cv2.imwrite(outname + 'mv.png', mv_b)
                        # cv2.imwrite('./mv_red/' + f'{pt_name}' + '.png', mv_b)
                        # cv2.imwrite('./mv_heat/' + f'{pt_name}' + '.png', mv_heat)
                        # cv2.imwrite('./sub_b_tot/' + "%.4f"% (dist) + '_' + f'{pt_name}' + "%.4f"% (error) + '_' + 'mv.png', tot)
                        
                        
                        # #warp by flow
                        # imgB = tensorBrgb.unsqueeze(dim = 0)
                        # mvHomo_f = get_warp_flow(imgB, homo_flow_f.unsqueeze(dim = 0)).squeeze(dim = 0).permute(1, 2, 0).cpu().numpy()
                        # hg_f = get_warp_flow(imgB, hg_flow_f.unsqueeze(dim = 0)).squeeze(dim = 0).permute(1, 2, 0).cpu().numpy()
                        # # mv_b = get_warp_flow(imgA, mv_flow.unsqueeze(dim = 0)).squeeze(dim = 0).permute(1, 2, 0).cpu().numpy()

                        # mvHomo_f[:,:,2] = imgArgb[:,:,2]
                        # hg_f[:,:,2] = imgArgb[:,:,2]
                        # # mv_b[:,:,2] = imgBrgb[:,:,2]
                        # tot = np.concatenate([mvHomo_f, hg_f], 1)

                        # outname = './sub_b/' + "%.4f"% (dist) + '_'  + f'{pt_name}' + "%.4f"% (error) + '_'
                        # cv2.imwrite(outname + 'codingHomo.png', mvHomo_f)
                        # cv2.imwrite(outname + 'HomoGAN.png', hg_f)
                        # # cv2.imwrite(outname + 'mv.png', mv_b)
                        # cv2.imwrite('./sub_b_tot/' + "%.4f"% (dist) + '_' + f'{pt_name}' + "%.4f"% (error) + '_' + 'mv.png', tot)

                    #cam mask
                    # for i in range(b):
                    #     tensorArgb, tensorBrgb = data_batch['imgs_full'][i, :3, ...], data_batch['imgs_full'][i, 3:, ...]
                    #     pt_name = data_batch["pt_names"][i]
                        # mask_b4 = output_batch['mask_b'][4][i,:,:,:]
                        # mask_f4 = output_batch['mask_f'][4][i,:,:,:]
                        # mask_vis_b = show_cam_on_image(tensorBrgb.permute(1, 2, 0).cpu().numpy()/255., mask_b4.permute(1, 2, 0).cpu().numpy())
                        # mask_vis_f = show_cam_on_image(tensorArgb.permute(1, 2, 0).cpu().numpy()/255., mask_f4.permute(1, 2, 0).cpu().numpy())
                        # outname = './vis_mask/' + f'{pt_name}' + '_'
                        # cv2.imwrite(outname + 'mask_b.png', mask_vis_b)
                        # cv2.imwrite(outname + 'mask_f.png', mask_vis_f)
                        
                        # homo_flow_b = output_batch["flow_b"][i,:,:,:].permute(2,0,1)

                        # name = pt_name.split('.npy')[0]

                        # img24 = cv2.imread('./CA_24/' + pt_name + '.jpg')
                        # img72 = cv2.imread('./CA_72/' + pt_name + '.jpg')

                        # # ten24 = torch.Tensor(img24).permute(2,0,1).unsqueeze(dim = 0).cuda()
                        # # ten72 = torch.Tensor(img72).permute(2,0,1).unsqueeze(dim = 0).cuda()

                        # # imgA = tensorArgb.unsqueeze(dim = 0)
                        # # w_ten24 = get_warp_flow(ten24, homo_flow_b.unsqueeze(dim = 0)).squeeze(dim = 0).permute(1, 2, 0).cpu().numpy()
                        # # w_ten72 = get_warp_flow(ten72, homo_flow_b.unsqueeze(dim = 0)).squeeze(dim = 0).permute(1, 2, 0).cpu().numpy()

                        # mask_vis24b = show_cam_on_image(tensorBrgb.permute(1, 2, 0).cpu().numpy()/255., img24/255.)
                        # mask_vis72b = show_cam_on_image(tensorBrgb.permute(1, 2, 0).cpu().numpy()/255., img72/255.)

                        # cv2.imwrite('./CA_mask_24/' + name + '.jpg', mask_vis24b)
                        # cv2.imwrite('./CA_mask_72/' + name + '.jpg', mask_vis72b)
                        




                    
                    #pyramid mask
                    # for i in range(b):
                    # #load form output
                    #     # tensorA, tensorB = data_batch["imgs_gray_full"][i, :1, ...], data_batch["imgs_gray_full"][i, 1:, ...]
                    #     tensorArgb, tensorBrgb = data_batch['imgs_full'][i, :3, ...], data_batch['imgs_full'][i, 3:, ...]
                    #     # homo_flow_f = output_batch["flow_f"][i,:,:,:].permute(2,0,1)
                    #     # hg_flow_b = data_batch['homo_flow_b'][i,:,:,:]
                    #     # hg_flow_f = data_batch['homo_flow_f'][i,:,:,:]
                    #     # _, h, w = tensorArgb.shape
                    #     # mv_flow = data_batch["mv_flow"][i,:,:,:]
                    #     # pts = data_batch["pt_set"]
                    #     pt_name = data_batch["pt_names"][i]
                    #     # error = err_avg[i]
                    #     # error_hg = eval_results["errors_hg"][i]
                    #     # dist = error - error_hg
                    #     homo_flow_b0 = output_batch["pyramid_flow_b"][0][i,:,:,:]
                    #     homo_flow_b1 = output_batch["pyramid_flow_b"][1][i,:,:,:]
                    #     homo_flow_b2 = output_batch["flow_b"][i,:,:,:].permute(2,0,1)
                    #     mask_b0 = output_batch['mask_b'][0][i,:,:,:]
                    #     mask_b1 = output_batch['mask_b'][1][i,:,:,:]
                    #     mask_b2 = output_batch['mask_b'][2][i,:,:,:]

                    #     imgBrgb = tensorBrgb.cpu().numpy().transpose(1,2,0)
                        
                    #     imgA = tensorArgb.unsqueeze(dim = 0)
                    #     mvHomo_b0 = get_warp_flow(imgA, homo_flow_b0.unsqueeze(dim = 0)).squeeze(dim = 0).permute(1, 2, 0).cpu().numpy()
                    #     mvHomo_b1 = get_warp_flow(imgA, homo_flow_b1.unsqueeze(dim = 0)).squeeze(dim = 0).permute(1, 2, 0).cpu().numpy()
                    #     mvHomo_b2 = get_warp_flow(imgA, homo_flow_b2.unsqueeze(dim = 0)).squeeze(dim = 0).permute(1, 2, 0).cpu().numpy()

                    #     mvHomo_heat_b0 = make_align_heatmap(mvHomo_b0, imgBrgb)
                    #     mvHomo_heat_b1 = make_align_heatmap(mvHomo_b1, imgBrgb)
                        # mvHomo_heat_b2 = make_align_heatmap(mvHomo_b2, imgBrgb)

                        # mask_vis_b0 = show_cam_on_image(tensorBrgb.permute(1, 2, 0).cpu().numpy()/255., mask_b0.permute(1, 2, 0).cpu().numpy())
                        # mask_vis_b1 = show_cam_on_image(imgBrgb/255., mask_b1.permute(1, 2, 0).cpu().numpy())
                        # mask_vis_b2 = show_cam_on_image(imgBrgb/255., mask_b2.permute(1, 2, 0).cpu().numpy())

                        # mvHomo_b0[:,:,2] = imgBrgb[:,:,2]
                        # mvHomo_b1[:,:,2] = imgBrgb[:,:,2]
                        # mvHomo_b2[:,:,2] = imgBrgb[:,:,2]

                        # red = np.concatenate([mvHomo_b0, mvHomo_b1, mvHomo_b2], 1)
                        # heat = np.concatenate([mvHomo_heat_b0, mvHomo_heat_b1, mvHomo_heat_b2], 1)
                        # mask = np.concatenate([mask_vis_b0, mask_vis_b1, mask_vis_b2], 1)

                        # out = np.concatenate([red, heat, mask], 0)

                        
                        # outname = './pyramid_mask/' + f'{pt_name}' + '_' + 'mask_b.jpg'
                        # cv2.imwrite(outname, out)

            MSE_RE_avg = sum(MSE_RE) / len(MSE_RE)
            MSE_LT_avg = sum(MSE_LT) / len(MSE_LT)
            MSE_LL_avg = sum(MSE_LL) / len(MSE_LL)
            MSE_SF_avg = sum(MSE_SF) / len(MSE_SF)
            MSE_LF_avg = sum(MSE_LF) / len(MSE_LF)
            MSE_avg = (MSE_RE_avg + MSE_LT_avg + MSE_LL_avg + MSE_SF_avg + MSE_LF_avg) / 5

            Metric = {
                "MSE_RE_avg": MSE_RE_avg,
                "MSE_LT_avg": MSE_LT_avg,
                "MSE_LL_avg": MSE_LL_avg,
                "MSE_SF_avg": MSE_SF_avg,
                "MSE_LF_avg": MSE_LF_avg,
                "AVG": MSE_avg
            }
            manager.update_metric_status(metrics=Metric, split="test")

            # update data to tensorboard
            manager.logger.info("Loss/valid epoch_val {}: {:.4f}. RE:{:.4f} LT:{:.4f} LL:{:.4f} SF:{:.4f} LF:{:.4f} ".format(
                manager.epoch_val, MSE_avg, MSE_RE_avg, MSE_LT_avg, MSE_LL_avg, MSE_SF_avg, MSE_LF_avg))

            manager.print_metrics("test", title="test", color="red")

            # for m in range(9):
            #     rubost_tot[m] = rubost_tot[m]/25200
                
            # print(rubost_tot)



if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    # Only load model weights
    params.only_weights = True

    # Update args into params
    params.update(vars(args))

    # Use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    logger = utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # Fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(params)

    # Define the model and optimizer
    if params.cuda:
        model = net.fetch_net(params).cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    else:
        model = net.fetch_net(params)

    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.gamma)

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(split_batches=True, kwargs_handlers=[kwargs])
    model, optimizer, dataloaders["train"], dataloaders["val"], dataloaders["test"], scheduler = accelerator.prepare(
        model, optimizer, dataloaders["train"], dataloaders["val"], dataloaders["test"], scheduler)

    # initial status for checkpoint manager
    manager = Manager(model=model,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      params=params,
                      dataloaders=dataloaders,
                      writer=None,
                      logger=logger,
                      accelerator=accelerator)

    # Initial status for checkpoint manager

    # Reload weights from the saved file
    manager.load_checkpoints()

    # Test the model
    logger.info("Starting test")

    # Evaluate
    evaluate(model, manager)
