import torch
import os
from skimage import img_as_ubyte

import argparse
import sys
sys.path.append(os.getcwd())
import np_metric as img_met
import metrics_glf_cr as metrics_glf_cr
import importlib
from os import path as osp

from torchvision.transforms import ToTensor

from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from metrics.psnr_ssim import calculate_mor_vor_cvor, calculate_mse, calculate_snr, equivalent_number_of_looks, speckle_suppression_and_mean_preservation_index, speckle_suppression_index
from basicsr.data.MiO_dataset import IRBenchmarks_test
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
from typing import List
import pathlib
from basicsr.utils.image_utils import save_img
from basicsr.utils.image_utils import tensor2img,intensity2normalizedAmp, normalizedAmp2intensity, imwrite, mergeimage,splitimage
metric_module = importlib.import_module('metrics')
from torch.utils.data import DataLoader
import logging
import datetime
# from basicsr.models.archs.CR_Former_Net import CR_former
# from basicsr.models.archs.TGP_Net import TGPNet
from basicsr.models.image_restoration_model import ImageCleanEPModel, ImageCleanModel
from basicsr.utils.sen_utils import GetQuadrupletsImg
from basicsr.utils.sen_utils import SaveImg

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
import numpy as np
logging.basicConfig(level = logging.CRITICAL,
    format = '%(asctime)s [%(levelname)s] at %(filename)s,%(lineno)d: %(message)s',
    datefmt = '%Y-%m-%d(%a)%H:%M:%S',
    filename = 'test_TGPNet_log_'+now+'.txt',
    filemode = 'w')


console = logging.StreamHandler()
console.setLevel(logging.CRITICAL)
formatter = logging.Formatter('[%(levelname)-8s] %(message)s') 
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

parser = argparse.ArgumentParser(description='Test on your own images')
parser.add_argument('--opt', default='option/TGPNet_test.yml', type=str, help='Path to config yml file')
# Input Parameters
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--data_file_dir', type=str, default="/data3/huang/datasets/AIO-RS/AIO-RS5/test/", help='Path to datasets.')
# parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--benchmarks', type=list, default=['decloud_rice1', 'decloud_rice2', 'deshadow', 'desar', 'denoise_15', 'denoise_25', 'denoise_50','deblurgauss_5', 'deblurgauss_10'], help='which benchmarks to test on.')
# parser.add_argument('--benchmarks', type=list, default=['decloud_rice1', 'decloud_rice2', 'deshadow', 'desar', 'denoise_15', 'denoise_25', 'denoise_50','deblurgauss_5', 'deblurgauss_10','decloud_sen12ms'], help='which benchmarks to test on.')
# parser.add_argument('--benchmarks', type=list, default=['decloud_sen12ms'], help='which benchmarks to test on.')
parser.add_argument('--de_type', type=list, default=['denoise_15', 'denoise_25', 'denoise_50', 'decloud', 'deshadow', 'desar','deblurgauss_5', 'deblurgauss_10','decloud_sen12ms'], help='Degradation types for training/testing.')
parser.add_argument('--save_results', action="store_true", help="Save restored outputs.")
testopt = parser.parse_args()
args = parser.parse_args()


####################################################################################################
## HELPERS
def compute_psnr(image_true, image_test, image_mask, data_range=None):
  # this function is based on skimage.metrics.peak_signal_noise_ratio
  err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
  return 10 * np.log10((data_range ** 2) / err)

def compute_ssim(tar_img, prd_img, cr1):
    ssim_pre, ssim_map = structural_similarity(tar_img, prd_img, channel_axis=2, gaussian_weights=True, data_range = 1.0, full=True)
    ssim_map = ssim_map * cr1
    r = int(3.5 * 1.5 + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    pad = (win_size - 1) // 2
    ssim = ssim_map[pad:-pad,pad:-pad,:]
    crop_cr1 = cr1[pad:-pad,pad:-pad,:]
    ssim = ssim.sum(axis=0).sum(axis=0)/crop_cr1.sum(axis=0).sum(axis=0)
    ssim = np.mean(ssim)
    return ssim

def calc_psnr(img1, img2, data_range=1.0):
    err = np.sum((img1 - img2) ** 2, dtype=np.float64)
    return 10 * np.log10((data_range ** 2) / (err / img1.size))

def calc_ssim(img1, img2,data_range = 1.0):
    return structural_similarity(img1, img2, channel_axis=2, gaussian_weights=True, data_range = data_range, full=False)


####################################################################################################
def run_test(opts, net, dataset, factor=8):
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, drop_last=False, num_workers=0)

    if opts.save_results:
        pathlib.Path(os.path.join(os.getcwd(), f"results/{opts.benchmarks[0]}")).mkdir(
            parents=True, exist_ok=True)
    calc_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True, reduction="mean").cuda()
    psnr, ssim, lpips = [], [], []

    with torch.no_grad():

        for data in tqdm(testloader):
            degrad_patch, clean_patch = data['lq'].cuda(), data['gt'].cuda()
            clean_name = data['lq_path']

            net.feed_data(data)
            net.test()
            restored = torch.clamp(net.output, 0, 1)

            # Forward pass
            # restored = net(degrad_patch)
            if isinstance(restored, List) and len(restored) == 2:
                restored, _ = restored

            # Unpad images to original dimensions
            assert restored.shape == clean_patch.shape, "Restored and clean patch shape mismatch."

            # restored = torch.clamp(restored, 0, 1)
            clean_patch = clean_patch[:,0:3]
            restored = restored[:,0:3]
            degrad_patch = degrad_patch[:,0:3]
            lpips.append(calc_lpips(clean_patch, restored).cpu().numpy())

            restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            degrad_patch = degrad_patch.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            clean = clean_patch.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            ssim.append(calc_ssim(clean, restored))
            psnr_temp = peak_signal_noise_ratio(clean, restored, data_range=1)
            psnr.append(psnr_temp)

            if opts.save_results:
                save_name = os.path.splitext(os.path.split(clean_name[0])[-1])[0] + '_' + str(
                    round(psnr_temp, 2)) + '.png'
                # save output images
                save_img(
                    (os.path.join(os.getcwd(),
                                  f"results/{opts.benchmarks[0]}",
                                  save_name)),
                    img_as_ubyte(restored))

                save_name_gt = os.path.splitext(os.path.split(clean_name[0])[-1])[0] + '_' + str(
                    round(psnr_temp, 2)) + '_gt.png'
                # save output images
                save_img(
                    (os.path.join(os.getcwd(),
                                  f"results/{opts.benchmarks[0]}",
                                  save_name_gt)),
                    img_as_ubyte(clean))

                save_name_de = os.path.splitext(os.path.split(clean_name[0])[-1])[0] + '_' + str(
                    round(psnr_temp, 2)) + '_degrade.png'
                # save output images
                save_img(
                    (os.path.join(os.getcwd(),
                                  f"results/{opts.benchmarks[0]}",
                                  save_name_de)),
                    img_as_ubyte(degrad_patch))

    print('PSNR: {:f} SSIM: {:f} LPIPS: {:f}\n'.format(np.mean(psnr), np.mean(ssim), np.mean(lpips)))
    logging.critical('PSNR: {:f} SSIM: {:f} LPIPS: {:f}\n'.format(np.mean(psnr), np.mean(ssim), np.mean(lpips)))


import math as m
####################################################################################################
def run_test_sen12ms(opts, net,dataset, factor=8):
    # 数据集
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, drop_last=False, num_workers=0)

    if opts.save_results:
        pathlib.Path(os.path.join(os.getcwd(), f"results/{opts.benchmarks[0]}")).mkdir(
            parents=True, exist_ok=True)
    output_dir = f"results/{opts.benchmarks[0]}"

    MAE_vs = []
    MSE_vs = []
    RMSE_vs = []
    BRMSE_vs = []
    ssim_vs = []
    psnr_vs = []
    sam_vs = []
    logging.critical("start testing...")

    with torch.no_grad():
        iteration = 0
        for data in tqdm(testloader):
            iteration += 1
            degrad_patch, clean_patch = data['lq'].cuda(), data['gt'].cuda()
            clean_name = data['lq_path']
            patch_path_out  = clean_name

            net.feed_data(data)
            net.test()
            restored = net.output

            # Forward pass
            # restored = net(degrad_patch)
            if isinstance(restored, List) and len(restored) == 2:
                restored, _ = restored

            # Unpad images to original dimensions
            assert restored.shape == clean_patch.shape, "Restored and clean patch shape mismatch."

            img_fake = restored[:, 2:]
            img_truth = clean_patch[:, 2:]

            output,img_cld_RGB,img_fake_RGB,img_truth_RGB,img_sar_RGB,img_cld_nobright_RGB = GetQuadrupletsImg(degrad_patch, img_fake, img_truth)

            if not os.path.exists(output_dir + '/test_img'):
                os.makedirs(output_dir + '/test_img')

            outfilename = patch_path_out[0].split('/')[-1].split('.')[0]
            # logging.critical(outfilename)
            SaveImg(output,
                    os.path.join(output_dir + '/test_img', "{}_{}_out.jpg".format(outfilename, iteration)))
            SaveImg(img_cld_RGB, os.path.join(output_dir + '/test_img',
                                              "{}_{}_incld_bright.jpg".format(outfilename, iteration)))
            SaveImg(img_fake_RGB,
                    os.path.join(output_dir + '/test_img', "{}_{}_outfake.jpg".format(outfilename, iteration)))
            SaveImg(img_truth_RGB,
                    os.path.join(output_dir + '/test_img', "{}_{}_truth.jpg".format(outfilename, iteration)))
            # SaveImg(img_csm_RGB,
            #         os.path.join(output_dir + '/test_img', "{}_{}_csm.jpg".format(outfilename, iteration)))
            SaveImg(img_sar_RGB,
                    os.path.join(output_dir + '/test_img', "{}_{}_sar.jpg".format(outfilename, iteration)))
            SaveImg(img_cld_nobright_RGB,
                    os.path.join(output_dir + '/test_img', "{}_{}_incld.jpg".format(outfilename, iteration)))

            s2img1 = img_truth.clone()
            fake_img1 = img_fake.clone() / 5  # convert values from 0-5 to 0-1
            s2img1 = s2img1 / 5
            MAE_v = img_met.cloud_mean_absolute_error(s2img1, fake_img1)
            MSE_v = img_met.cloud_mean_squared_error(s2img1, fake_img1)
            RMSE_v = img_met.cloud_root_mean_squared_error(s2img1, fake_img1)
            BRMSE_v = img_met.cloud_bandwise_root_mean_squared_error(s2img1, fake_img1)
            # ssim_v = img_met.cloud_ssim(s2img1, fake_img1)
            # psnr_v = img_met.cloud_psnr(s2img1, fake_img1) # equal to glf_cr.psnr
            psnr_v = metrics_glf_cr.PSNR(s2img1, fake_img1)
            ssim_v = metrics_glf_cr.SSIM(s2img1, fake_img1)

            MAE_vs.append(np.asarray(MAE_v.cpu()))
            MSE_vs.append(np.asarray(MSE_v.cpu()))
            RMSE_vs.append(np.asarray(RMSE_v.cpu()))
            BRMSE_vs.append(np.asarray(BRMSE_v.cpu()))
            ssim_vs.append(np.asarray(ssim_v.cpu().detach().numpy()))
            psnr_vs.append(np.asarray(psnr_v))

            # spectral angle mapper
            mat = s2img1 * fake_img1
            mat = torch.sum(mat, 1)
            mat = torch.div(mat, torch.sqrt(torch.sum(s2img1 * s2img1, 1)))
            mat = torch.div(mat, torch.sqrt(torch.sum(fake_img1 * fake_img1, 1)))
            sam_v = torch.mean(torch.acos(torch.clamp(mat, -1, 1)) * torch.tensor(180) / m.pi)
            sam_vs.append(np.asarray(sam_v.cpu().detach().numpy()))

            # logging.critical(
            #     "MAE_v:{:.6f},MSE_v:{:.6f},RMSE_v:{:.6f},BRMSE_v:{:.6f},psnr_v:{:.6f},ssim_v:{:.6f},sam_v:{:.6f}".format(
            #         MAE_v.cpu(), MSE_v.cpu(),
            #         RMSE_v.cpu(), BRMSE_v.cpu(),
            #         psnr_v, ssim_v.cpu(), sam_v))

        MAE_v = np.mean(MAE_vs)
        MSE_v = np.mean(MSE_vs)
        RMSE_v = np.mean(RMSE_vs)
        BRMSE_v = np.mean(BRMSE_vs)
        ssim_v = np.mean(ssim_vs)
        psnr_v = np.mean(psnr_vs)
        sam_v = np.mean(sam_vs)
        # print("MAE_v:{:.6f},MSE_v:{:.6f},RMSE_v:{:.6f},BRMSE_v:{:.6f},psnr_v:{:.6f}".format(MAE_v, MSE_v, RMSE_v, BRMSE_v,
        #                                                                          psnr_v))
        print(
            "MAE_m:{:.6f},MSE_m:{:.6f},RMSE_m:{:.6f},BRMSE_m:{:.6f},psnr_m:{:.6f},ssim_m:{:.6f},sam_m:{:.6f}".format(
                MAE_v, MSE_v, RMSE_v, BRMSE_v,
                psnr_v, ssim_v, sam_v))
        logging.critical(
            "MAE_m:{:.6f},MSE_m:{:.6f},RMSE_m:{:.6f},BRMSE_m:{:.6f},psnr_m:{:.6f},ssim_m:{:.6f},sam_m:{:.6f}".format(
                MAE_v, MSE_v, RMSE_v, BRMSE_v,
                psnr_v, ssim_v, sam_v))


from torchvision import transforms
from collections import OrderedDict

def get_current_visuals(lq, output, gt):
    out_dict = OrderedDict()
    out_dict['lq'] = lq.detach().cpu()
    out_dict['result'] = output.detach().cpu()
    out_dict['gt'] = gt.detach().cpu()

    out_dict['lq'] = out_dict['lq'].mean(dim=1, keepdim=True)
    out_dict['gt'] = out_dict['gt'].mean(dim=1, keepdim=True)

    gray_transform = transforms.Grayscale(num_output_channels=1)
    out_dict['result'] = out_dict['result'].mean(dim=1, keepdim=True)  # prcesion high, use average
    # gray_pil = gray_transform(out_dict['result'])  #low
    # out_dict['result'] = gray_pil

    return out_dict


def run_sar_test(opts, net, dataset, factor=8):
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, drop_last=False, num_workers=0)

    rgb2bgr = True
    use_image = True
    save_img_bool = True
    with_metrics = True

    if opts.save_results:
        pathlib.Path(os.path.join(os.getcwd(), f"results/{opts.benchmarks[0]}/val")).mkdir(
            parents=True, exist_ok=True)
    calc_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True, reduction="mean").cuda()

    psnr, ssim, lpips = [], [], []

    metric_results = {}
    for sar_metric in ['psnr', 'ssim', 'MOR', 'CVOR', 'SNR', 'MSE', 'ssim_view', 'psnr_view']:
        metric_results[sar_metric] = 0

    with torch.no_grad():
        cnt = 0
        # for ([clean_name, de_id], degrad_patch, clean_patch, nrd_lq_min, nrd_lq_max) in tqdm(testloader):
        for data in tqdm(testloader):
            degrad_patch, clean_patch = data['lq'].cuda(), data['gt'].cuda()
            clean_name = data['lq_path']
            nrd_lq_min = data['nrd_lq_min']
            nrd_lq_max= data['nrd_lq_max']
            # degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            img_name = os.path.splitext(os.path.split(clean_name[0])[-1])[0]

            net.feed_data(data)
            net.test()
            restored = net.output

            # Forward pass
            # restored = net(degrad_patch)
            if isinstance(restored, List) and len(restored) == 2:
                restored, _ = restored

            # Unpad images to original dimensions
            assert restored.shape == clean_patch.shape, "Restored and clean patch shape mismatch."

            ###start cl-sar-Despeckling
            clean_patch = clean_patch[:,0:3]
            restored = restored[:,0:3]
            degrad_patch = degrad_patch[:,0:3]
            visuals = get_current_visuals(degrad_patch, restored, clean_patch)

            lq_min, lq_max = nrd_lq_min.cpu().numpy(), nrd_lq_max.cpu().numpy()
            sr_img_unnormalized = tensor2img([visuals['result']], rgb2bgr=rgb2bgr, out_type=np.float32)
            lq_img_intensity = tensor2img([visuals['lq']], rgb2bgr=rgb2bgr, out_type=np.float32)
            gt_img_intensity = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr, out_type=np.float32)

            # real SAR
            sr_img_intensity = normalizedAmp2intensity(sr_img_unnormalized, lq_min, lq_max)
            lq_img_intensity = normalizedAmp2intensity(lq_img_intensity, lq_min, lq_max)  # [B, C, H, W] -> **2 -> H,W
            gt_img_intensity = normalizedAmp2intensity(gt_img_intensity, lq_min, lq_max)  # intensity

            lq_img = intensity2normalizedAmp(lq_img_intensity)  # [0, 1]
            sr_img = intensity2normalizedAmp(sr_img_intensity)
            gt_img = intensity2normalizedAmp(gt_img_intensity)

            if opts.save_results:
                save_name = os.path.splitext(os.path.split(clean_name[0])[-1])[0]
                save_img_path = os.path.join(os.getcwd(),
                                             f"results/{opts.benchmarks[0]}/val",
                                             f'{save_name}_sr.png')
                save_gt_img_path = os.path.join(os.getcwd(),
                                                f"results/{opts.benchmarks[0]}/val",
                                                f'{save_name}_gt.png')
                save_lq_img_path = os.path.join(os.getcwd(),
                                                f"results/{opts.benchmarks[0]}/val",
                                                f'{save_name}_lq.png')

                def _view(img):
                    _sum = np.sum(img)
                    _len = len(np.nonzero(img)[0])
                    scale = 0.3 / (_sum / _len)
                    img = img * scale
                    img = np.where(img > 1, 1, img)
                    img = (img * 255.).astype(np.uint8)
                    return img

                lq_img_save = _view(lq_img)
                sr_img_save = _view(sr_img)
                gt_img_save = _view(gt_img)

                imwrite(sr_img_save, save_img_path)
                imwrite(gt_img_save, save_gt_img_path)
                imwrite(lq_img_save, save_lq_img_path)

                # concat = np.concatenate((lq_img_save, sr_img_save, gt_img_save), axis=1)[:, :, np.newaxis]

                # calculate ssim and psnr for view
                psnr_view = peak_signal_noise_ratio(sr_img_save, gt_img_save, data_range=255)
                ssim_view = structural_similarity(sr_img_save, gt_img_save, data_range=255)
                metric_results['psnr_view'] += psnr_view
                metric_results['ssim_view'] += ssim_view

            # save intensity numpy for compare
            save_npy_dir = os.path.join(os.getcwd(),
                                        f"results/{opts.benchmarks[0]}/val",
                                        'eval_I')
            save_npy_path = osp.join(save_npy_dir, f'{img_name}_I.npy')
            os.makedirs(save_npy_dir, exist_ok=True)
            np.save(save_npy_path, sr_img_intensity)

            if with_metrics:
                # calculate metrics
                opt_metric = OrderedDict()
                opt_metric['psnr'] = OrderedDict()
                opt_metric['psnr']['type'] = 'calculate_psnr'
                opt_metric['psnr']['crop_border'] = 0
                opt_metric['psnr']['test_y_channel'] = False

                opt_metric['ssim'] = OrderedDict()
                opt_metric['ssim']['type'] = 'calculate_ssim'
                opt_metric['ssim']['crop_border'] = 0
                opt_metric['ssim']['test_y_channel'] = False

                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)

                    mor, _, cvor = calculate_mor_vor_cvor(sr_img_intensity, lq_img_intensity)
                    metric_results['MOR'] += mor
                    metric_results['CVOR'] += cvor
                    snr = calculate_snr(sr_img_intensity, lq_img_intensity)
                    metric_results['SNR'] += snr
                    # enl = equivalent_number_of_looks(sr_img_intensity, lq_img_intensity)
                    # self.metric_results['ENL'] += enl
                    mse = calculate_mse(sr_img, gt_img)
                    metric_results['MSE'] += mse
                else:
                    raise NotImplementedError
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

            ###end cl-sar-Despeckling

            # save output images
            # restored = torch.clamp(restored, 0, 1)
            # clean_patch = torch.clamp(clean_patch, 0, 1)
            # lpips.append(calc_lpips(clean_patch, restored).cpu().numpy())
            #
            # restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            # degrad_patch = degrad_patch.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            # clean = clean_patch.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            clean = gt_img
            restored = sr_img

            clean_new = clean[:, :, np.newaxis]
            clean_3 = np.repeat(clean_new, 3, axis=2)
            restored_new = restored[:, :, np.newaxis]
            restored_3 = np.repeat(restored_new, 3, axis=2)

            lpips.append(calc_lpips(ToTensor()(clean_3).cuda().unsqueeze(dim=0),
                                    ToTensor()(restored_3).cuda().unsqueeze(dim=0)).cpu().numpy())
            ssim.append(structural_similarity(clean, restored, data_range=1))
            psnr_temp = peak_signal_noise_ratio(clean, restored, data_range=1)
            psnr.append(psnr_temp)

            # if opts.save_results:
            #     save_name = os.path.splitext(os.path.split(clean_name[0])[-1])[0] + '_' + str(
            #         round(psnr_temp, 2)) + '.png'
            #     save_img(
            #         (os.path.join(os.getcwd(),
            #                       f"results/{opts.checkpoint_id}/{opts.benchmarks[0]}",
            #                       save_name)),
            #         img_as_ubyte(restored))

        current_metric_all = ''
        if with_metrics:
            for metric in metric_results.keys():
                metric_results[metric] /= cnt
                current_metric = metric_results[metric]

                current_metric_all += metric + ': {:f}, '.format(current_metric)
            print(current_metric_all)

    print('0-1 img PSNR: {:f} SSIM: {:f} LPIPS: {:f}\n'.format(np.mean(psnr), np.mean(ssim), np.mean(lpips)))
    logging.critical('0-1 img PSNR: {:f} SSIM: {:f} LPIPS: {:f} PSNR_view: {:f}, SSIM_view: {:f}\n'.format(np.mean(psnr),
                                                                                                         np.mean(ssim),
                                                                                                         np.mean(lpips),
                                                                                                         metric_results[
                                                                                                             'psnr_view'],
                                                                                                         metric_results[
                                                                                                             'ssim_view']))

    # logging.critical('0-1 img PSNR_view: '+ str(metric_results['psnr_view'] ) )#+ 'SSIM_view: {:f}\n'.format(metric_results['ssim_view']))

## test decloud rice1
def run_decloud_rice1(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)


## test decloud rice2
def run_decloud_rice2(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)


## test decloud tcloud
def run_decloud_tcloud(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)


## test deshadow
def run_deshadow(opts, net, dataset, factor=8):
    # run_deshadow_test(opts, net, dataset, factor)
    run_test(opts, net, dataset, factor)


## test deshadow
def run_desar(opts, net, dataset, factor=8):
    run_sar_test(opts, net, dataset, factor)


## test Dehaze
def run_dehaze(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)


## test synthetic denoising
def run_denoise_15(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)


def run_denoise_25(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)


def run_denoise_50(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)

## test synthetic denoising
def run_deblurgauss_5(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)

## test synthetic denoising
def run_deblurgauss_10(opts, net, dataset, factor=8):
    run_test(opts, net, dataset, factor)

def run_decloud_sen12ms(opts, net, dataset,factor=8):
    run_test_sen12ms(opts, net, dataset, factor)

# out_dir = os.path.join(args.result_dir)
#
# os.makedirs(out_dir, exist_ok=True)

# Get model weights and parameters
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

yaml_file= args.opt

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

# s = x['network_g'].pop('type')

x['is_train'] = False
x['dist'] = False

#model = CR_former(**x['network_g'])
#model = ARSIR_EP(**x['network_g'])
model = ImageCleanEPModel(x)
#model = ImageCleanModel(x)

#checkpoint = torch.load(args.weights)
#model.net_g.load_state_dict(checkpoint['params'])
# model.cuda()

# model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for de in testopt.benchmarks:
    ind_opt = testopt
    ind_opt.benchmarks = [de]

    dataset = IRBenchmarks_test(ind_opt)

    print("--------> Testing on", de, "testset.")
    # logging.CRITICAL("--------> Testing on "+ de+ " testset.")
    logging.critical("--------> Testing on %s testset.", de)

    print("\n")
    globals()[f"run_{de}"](testopt, model, dataset,factor=8)


def depth_type(value):
    try:
        return int(value)  # Try to convert to int
    except ValueError:
        return value  # If it fails, return the string


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

