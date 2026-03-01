import os
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.data.degradation_utils import Degradation
from basicsr.utils.image_utils import random_augmentation, crop_img, max_normalize, sar_augment,view_sar,img2tensor, add_noise
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor, Resize, InterpolationMode
import numpy as np
import torch
import glob
import random
from PIL import Image, ImageFilter
import cv2
import time
from .sen12ms_cr_dataset import SEN12MSCRDataset
import numpy as np
from torch.utils import data
import os
from enum import Enum
# import rasterio
# from .feature_detectors import get_cloud_cloudshadow_mask

class AIOSR_SAR_EP_TrainDataset(data.Dataset):
    """
    Dataset class for training on degraded images.
    """

    def __init__(self, args):
        super(AIOSR_SAR_EP_TrainDataset, self).__init__()
        self.opt = args
        self.de_temp = 0
        self.de_type = self.opt['de_type']
        self.D = Degradation(args)
        self.de_dict = {dataset: idx for idx, dataset in enumerate(self.de_type)}
        self.de_dict_reverse = {idx: dataset for idx, dataset in enumerate(self.de_type)}

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args['gt_size']),
        ])
        self.resize_transform = Compose([
            ToPILImage(),  # Converts tensor to PIL
            Resize((args['gt_size'], args['gt_size'])),
        ])

        self.toTensor = ToTensor()

        self._init_lr()
        self._merge_tasks()
        self.patch_size = args['gt_size']

        # sar
        self.is_view_to_gray = False

        type_npl = np.load('./data/type7.npy').astype(np.float32)
        self.typep = torch.from_numpy(type_npl).clone()  # ncsa 012345

    def __getitem__(self, idx):
        lr_sample = self.lr[idx]
        # print(lr_sample["img"])
        # print("1111111111")

        de_id = lr_sample["de_type"]
        deg_type = self.de_dict_reverse[de_id]
        img_typee = self.typep[0]
        clean_name = lr_sample["img"]

        if deg_type == "denoise_15" or deg_type == "denoise_25" or deg_type == "denoise_50":
            img_typee = self.typep[0]

            hr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16, min_width=self.patch_size)

            hr = self.crop_transform(hr)
            hr = np.array(hr)

            hr = random_augmentation(hr)[0]

            if deg_type == "denoise_15":
                lr = self.D.single_degrade(hr, 0)
            elif deg_type == "denoise_25":
                lr = self.D.single_degrade(hr, 1)
            elif deg_type == "denoise_50":
                lr = self.D.single_degrade(hr, 2)

            hr = np.concatenate((hr, hr, hr, hr, hr), axis=2)
            lr = np.concatenate((lr, lr, lr, lr, lr), axis=2)

        elif deg_type == "deblurgauss_5" or deg_type == "deblurgauss_10":
            img_typee = self.typep[4]
            hr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16, min_width=self.patch_size)
            hr = self.crop_transform(hr)
            hr = np.array(hr)

            hr = random_augmentation(hr)[0]
            if deg_type == "deblurgauss_5":
                lr = self.D.single_degrade(hr, 6)
            elif deg_type == "deblurgauss_10":
                lr = self.D.single_degrade(hr, 7)

            hr = np.concatenate((hr, hr, hr, hr, hr), axis=2)
            lr = np.concatenate((lr, lr, lr, lr, lr), axis=2)

        else:
            if deg_type == "dehaze":
                lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16, min_width=self.patch_size)
                clean_name = self._get_nonhazy_name(lr_sample["img"])
                hr = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16, min_width=self.patch_size)
                lr, hr = random_augmentation(*self._crop_patch(lr, hr))

                hr = np.concatenate((hr, hr, hr, hr, hr), axis=2)
                lr = np.concatenate((lr, lr, lr, lr, lr), axis=2)
            elif deg_type == "decloud":
                img_typee = self.typep[1]
                lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16, min_width=self.patch_size)
                clean_name = self._get_noncloud_name(lr_sample["img"])
                hr = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16, min_width=self.patch_size)
                lr, hr = random_augmentation(*self._crop_patch(lr, hr))

                hr = np.concatenate((hr, hr, hr, hr, hr), axis=2)
                lr = np.concatenate((lr, lr, lr, lr, lr), axis=2)
            elif deg_type == "decloud_tcloud":
                img_typee = self.typep[1]
                lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16, min_width=self.patch_size)
                clean_name = self._get_noncloud_name(lr_sample["img"])
                hr = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16, min_width=self.patch_size)
                lr, hr = random_augmentation(*self._crop_patch(lr, hr))

                hr = np.concatenate((hr, hr, hr, hr, hr), axis=2)
                lr = np.concatenate((lr, lr, lr, lr, lr), axis=2)
            elif deg_type == "deshadow":
                img_typee = self.typep[2]
                lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16, min_width=self.patch_size)
                clean_name = self._get_nonshadow_name(lr_sample["img"])
                hr = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16, min_width=self.patch_size)
                lr, hr = random_augmentation(*self._crop_patch(lr, hr))

                hr = np.concatenate((hr, hr, hr, hr, hr), axis=2)
                lr = np.concatenate((lr, lr, lr, lr, lr), axis=2)
            elif deg_type == "desar":
                img_typee = self.typep[3]
                # hr_sample = self.hr_sample[idx]
                if lr_sample["img"].endswith(".npy"):
                    lr, hr = self.get_sar_iterm(lr_sample["img"])
                elif lr_sample["img"].endswith(".tif"):
                    lr, hr = self.get_sar_syc_iterm(lr_sample["img"])

                hr = np.concatenate((hr, hr, hr, hr, hr), axis=2)
                lr = np.concatenate((lr, lr, lr, lr, lr), axis=2)
            elif deg_type == "decloud_sen12ms":
                img_typee = self.typep[5]
                lr, _, hr, clean_name = self.sen12dataset.getsen12ms_item(clean_name) #lr（15,256,256）, hr 13,256,256
                lr = np.transpose(lr, (1, 2, 0))
                hr = np.transpose(hr, (1, 2, 0))
                lr = crop_img(lr, base=16, min_width=self.patch_size) #h,w,c
                hr = crop_img(hr, base=16, min_width=self.patch_size)
                lr, hr = random_augmentation(*self._crop_patch(lr, hr))
                lr_sample["img"] = clean_name

            else:
                hr_sample = self.hr[idx]
                lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16, min_width=self.patch_size)
                hr = crop_img(np.array(Image.open(hr_sample["img"]).convert('RGB')), base=16, min_width=self.patch_size)
                lr, hr = random_augmentation(*self._crop_patch(lr, hr))


        lr = self.toTensor(lr)
        hr = self.toTensor(hr)

        # return [lr_sample["img"], de_id], lr, hr, img_typee

        return {'lq': lr, 'gt': hr, 'lq_path': lr_sample["img"], 'gt_path': clean_name,'img_type':img_typee,'img_type_word':deg_type}


    def get_sar_iterm(self,lr_sample_path):
        nrd_lq_path = lr_sample_path
        # nrd_lq_path = hr_sample["img"]
        nrd_gt_path = self._get_nonsar_name(lr_sample_path)
        # print(nrd_gt_path)
        # print(nrd_lq_path)
        nrd_ori_gt = np.load(nrd_gt_path)
        nrd_ori_lq = np.load(nrd_lq_path)  # H,W

        if len(nrd_ori_lq.shape) == 2:
            nrd_ori_lq = nrd_ori_lq[:, :, np.newaxis]
            nrd_ori_gt = nrd_ori_gt[:, :, np.newaxis]
        # nrd_ori_gt, nrd_ori_lq = paired_random_crop(nrd_ori_gt, nrd_ori_lq, self.crop_pad_size, 1,
        #                                             gt_path=nrd_gt_path)
        lr = crop_img(nrd_ori_lq, base=16, min_width=self.patch_size)
        hr = crop_img(nrd_ori_gt, base=16, min_width=self.patch_size)

        lr, hr = random_augmentation(*self._crop_patch(lr, hr))
        # intensity -> amplitude
        nrd_gt = np.sqrt(hr)
        nrd_lq = np.sqrt(lr)

        if self.is_view_to_gray:
            nrd_gt, _, _ = max_normalize(nrd_gt)  # amp to norm
            nrd_lq, _, _ = max_normalize(nrd_lq)
            nrd_gt = view_sar(nrd_gt)  # norm to gray
            nrd_lq = view_sar(nrd_lq)

        nrd_lq, nrd_lq_min, nrd_lq_max = max_normalize(nrd_lq)  # [0,1]
        nrd_gt_min, nrd_gt_max = nrd_lq_min, nrd_lq_max

        nrd_gt[np.isnan(nrd_gt)] = 0
        nrd_gt = np.abs((nrd_gt - nrd_lq_min) / (nrd_lq_max - nrd_lq_min))
        lr = nrd_lq
        hr = nrd_gt
        lr = np.repeat(lr, 3, axis=2)
        hr = np.repeat(hr, 3, axis=2)

        return lr, hr

    def get_sar_syc_iterm(self, lr_sample_path):
        sd_ori_gt = cv2.imread(lr_sample_path)
        sd_ori_gt = cv2.cvtColor(sd_ori_gt, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sd_ori_lq = add_noise(sd_ori_gt).astype(np.float32)  # L=1 is too speckled
        if len(sd_ori_lq.shape) == 2:
            sd_ori_lq = sd_ori_lq[:, :, np.newaxis]
            sd_ori_gt = sd_ori_gt[:, :, np.newaxis]
        # sd_ori_gt, sd_ori_lq = paired_random_crop(sd_ori_gt, sd_ori_lq, self.patch_size, 1, gt_path=lr_sample_path)
        lr = crop_img(sd_ori_lq, base=16, min_width=self.patch_size)
        hr = crop_img(sd_ori_gt, base=16, min_width=self.patch_size)

        lr, hr = random_augmentation(*self._crop_patch(lr, hr))

        lr, sd_lq_min, sd_lq_max = max_normalize(lr)
        sd_gt_min, sd_gt_max = sd_lq_min, sd_lq_max
        hr[np.isnan(hr)] = 0
        hr = np.abs((hr - sd_lq_min) / (sd_lq_max - sd_lq_min))
        hr[np.isnan(hr)] = 0

        lr = np.repeat(lr, 3, axis=2)
        hr = np.repeat(hr, 3, axis=2)

        return lr, hr

    def __len__(self):
        return len(self.lr)

    def _init_lr(self):
        # synthetic datasets
        if 'denoise_15' in self.de_type:
            self._init_clean(id=self.de_dict['denoise_15'])
        if 'denoise_25' in self.de_type:
            self._init_clean(id=self.de_dict['denoise_25'])
        if 'denoise_50' in self.de_type:
            self._init_clean(id=self.de_dict['denoise_50'])
        if 'decloud' in self.de_type:
            self._init_decloud(self.de_dict['decloud'])
        if 'deshadow' in self.de_type:
            self._init_deshadow(self.de_dict['deshadow'])
        if 'desar' in self.de_type:
            self._init_desar(self.de_dict['desar'])
        if 'decloud_tcloud' in self.de_type:
            self._init_decloud_tcloud(self.de_dict['decloud_tcloud'])
        if 'deblurgauss_5' in self.de_type:
            self._init_blurgauss(id=self.de_dict['deblurgauss_5'])
        if 'deblurgauss_10' in self.de_type:
            self._init_blurgauss(id=self.de_dict['deblurgauss_10'])
        if 'decloud_sen12ms' in self.de_type:
            self._init_decloud_sen12ms(self.de_dict['decloud_sen12ms'])

    def _merge_tasks(self):
        self.lr = []
        self.hr = []

        if "denoise_15" in self.de_type:
            self.lr += self.s15_ids
            self.hr += self.s15_ids
        if "denoise_25" in self.de_type:
            self.lr += self.s25_ids
            self.hr += self.s25_ids
        if "denoise_50" in self.de_type:
            self.lr += self.s50_ids
            self.hr += self.s50_ids
        if "decloud" in self.de_type:
            self.lr += self.decloud_lr
            self.hr += self.decloud_lr
        if "deshadow" in self.de_type:
            self.lr += self.deshadow_lr
            self.hr += self.deshadow_lr
        if "desar" in self.de_type:
            self.lr += self.desar_lr
            self.hr += self.desar_lr
        if "decloud_tcloud" in self.de_type:
            self.lr += self.decloud_tcloud_lr
            self.hr += self.decloud_tcloud_lr
        if "deblurgauss_5" in self.de_type:
            self.lr += self.blur5_ids
            self.hr += self.blur5_ids
        if "deblurgauss_10" in self.de_type:
            self.lr += self.blur10_ids
            self.hr += self.blur10_ids
        self.lr = self.lr * 10
        self.hr = self.hr * 10
        # print("4task*10: " + str(len(self.lr)))
        if "decloud_sen12ms" in self.de_type:
            self.lr += self.decloud_sen12ms_lr
            self.hr += self.decloud_sen12ms_lr

        print("5task total: " + str(len(self.lr)))

    def _init_deblur(self, id):
        """ Initialize the GoPro training dataset """
        inputs = self.opt['data_file_dir'] + "/deblurring/GoPro/crop/train/input_crops/"
        targets = self.opt['data_file_dir'] + "/deblurring/GoPro/crop/train/target_crops/"

        self.deblur_lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.deblur_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.png"))]

        self.deblur_counter = 0
        print("Total Deblur training pairs : {}".format(len(self.deblur_hr)))
        self.deblur_lr = self.deblur_lr * 5
        self.deblur_hr = self.deblur_hr * 5
        print("Repeated Dataset length : {}".format(len(self.deblur_hr)))

    def _init_clean(self, id):
        inputs = self.opt['data_file_dir'] + "/denoising"

        clean = []
        for dataset in ["UCMerced_LandUse"]:
            if dataset == "UCMerced_LandUse":
                ext = "tif"
            else:
                ext = "jpg"
            clean += [x for x in sorted(glob.glob(inputs + f"/{dataset}/*.{ext}"))]

        try:
            if id == self.de_dict['denoise_15']:  # 'denoise_15' in self.de_type:
                self.s15_ids = [{"img": x, "de_type": self.de_dict['denoise_15']} for x in clean]
                self.s15_ids = self.s15_ids * 3
                random.shuffle(self.s15_ids)
                self.s15_counter = 0
            if id == self.de_dict['denoise_25']:  # 'denoise_25' in self.de_type:
                self.s25_ids = [{"img": x, "de_type": self.de_dict['denoise_25']} for x in clean]
                self.s25_ids = self.s25_ids * 3
                random.shuffle(self.s25_ids)
                self.s25_counter = 0
            if id == self.de_dict['denoise_50']:  # 'denoise_50' # in self.de_type:
                self.s50_ids = [{"img": x, "de_type": self.de_dict['denoise_50']} for x in clean]
                self.s50_ids = self.s50_ids * 3
                random.shuffle(self.s50_ids)
                self.s50_counter = 0
                print("Repeated Denoise Ids : {}".format(len(self.s50_ids)))
        except Exception as e:
            print(f"read blurgauss error: {e}")

        self.num_clean = len(clean)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _init_blurgauss(self, id):
        inputs = self.opt['data_file_dir'] + "/deblur/hit-uav-DatasetNinja/img"

        clean = []
        ext = "jpg"
        clean += [x for x in sorted(glob.glob(inputs + f"/*.{ext}"))]
        try:
            if id == self.de_dict['deblurgauss_5']:  # 'deblurgauss_5' in self.de_type:
                self.blur5_ids = [{"img": x, "de_type": self.de_dict['deblurgauss_5']} for x in clean]
                self.blur5_ids = self.blur5_ids * 3
                random.shuffle(self.blur5_ids)
                self.blur5_counter = 0
                print("Repeated deblurgauss Ids : {}".format(len(self.blur5_ids)))
            if id == self.de_dict['deblurgauss_10']:  # 'deblurgauss_10' in self.de_type:
                self.blur10_ids = [{"img": x, "de_type": self.de_dict['deblurgauss_10']} for x in clean]
                self.blur10_ids = self.blur10_ids * 3
                random.shuffle(self.blur10_ids)
                self.blur10_counter = 0
                print("Repeated deblurgauss Ids : {}".format(len(self.blur10_ids)))

        except Exception as e:
            print(f"read blurgauss error: {e}")


        self.num_blurclean = len(clean)
        print("Total deblurgauss Ids : {}".format(self.num_blurclean))

    def _init_decloud(self, id):
        """ Initialize the GoPro training dataset """
        inputs = self.opt['data_file_dir'] + "/decloud/rice1/cloud/"
        targets = self.opt['data_file_dir'] + "/decloud/rice1/reference/"
        decloud_rice1_lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.png"))]
        # decloud_rice1_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.png"))]

        inputs2 = self.opt['data_file_dir'] + "/decloud/rice2/cloud/"
        targets2 = self.opt['data_file_dir'] + "/decloud/rice2/reference/"
        decloud_rice2_lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs2 + "/*.png"))]
        # decloud_rice2_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets2 + "/*.png"))]


        self.decloud_lr = decloud_rice1_lr + decloud_rice2_lr
        # self.decloud_hr = decloud_rice1_hr + decloud_rice2_hr

        self.decloud_counter = 0
        print("Total Decloud training pairs : {}".format(len(self.decloud_lr)))
        self.decloud_lr = self.decloud_lr * 5
        # self.decloud_hr = self.decloud_hr * 5
        random.shuffle(self.decloud_lr)
        print("Repeated Decloud Dataset length : {}".format(len(self.decloud_lr)))

    def _init_decloud_tcloud(self, id):
        """ Initialize the GoPro training dataset """
        inputs = self.opt['data_file_dir'] + "/decloud/T-Cloud/cloud/"
        targets = self.opt['data_file_dir'] + "/decloud/T-Cloud/reference/"
        decloud_tcloud_lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.png"))]
        # decloud_tcloud_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.png"))]

        self.decloud_tcloud_lr = decloud_tcloud_lr
        # self.decloud_tcloud_hr = decloud_tcloud_hr

        self.decloud_tcloud_counter = 0
        print("Total Decloud Tcloud training pairs : {}".format(len(self.decloud_tcloud_lr)))
        self.decloud_tcloud_lr = self.decloud_tcloud_lr * 5
        # self.decloud_tcloud_hr = self.decloud_tcloud_hr * 5
        random.shuffle(self.decloud_tcloud_lr)
        print("Repeated Decloud Tcloud Dataset length : {}".format(len(self.decloud_tcloud_lr)))

    def _init_decloud_sen12ms(self, id):
        """ Initialize the sen12ms training dataset """
        #train_datset_dir = "/data3/huang/datasets/AIO-RS/AIO-RS4/test/decloud/SEN12MS-CR"
        train_datset_dir = self.opt['data_file_dir'] + "/decloud/SEN12MS-CR"
        self.sen12dataset = SEN12MSCRDataset(train_datset_dir)

        decloud_sen12ms_lr = [{"img": x, "de_type": id} for x in self.sen12dataset.img]

        self.decloud_sen12ms_lr = decloud_sen12ms_lr

        self.decloud_sen12ms_counter = 0
        print("Total Decloud sen12ms training pairs : {}".format(len(self.decloud_sen12ms_lr)))
        # print("Repeated Decloud sen12ms Dataset length : {}".format(len(self.decloud_sen12ms_lr)))


    def _init_deshadow(self, id):
        """ Initialize the GoPro training dataset """
        inputs = self.opt['data_file_dir'] + "/deshadow/SRD/shadow/"
        targets = self.opt['data_file_dir'] + "/deshadow/SRD/shadow_free/"
        self.deshadow_lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.jpg"))]
        # self.deshadow_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.jpg"))]

        self.deshadow_counter = 0
        print("Total Deshadow training pairs : {}".format(len(self.deshadow_lr)))
        self.deshadow_lr = self.deshadow_lr * 2
        # self.deshadow_hr = self.deshadow_hr * 2
        random.shuffle(self.deshadow_lr)
        print("Repeated Deshadow Dataset length : {}".format(len(self.deshadow_lr)))

    def _init_desar(self, id):
        """ Initialize the GoPro training dataset """
        inputs = self.opt['data_file_dir'] + "/despeckling/SARdata-512/ori/"
        targets = self.opt['data_file_dir'] + "/despeckling/SARdata-512/model_1_org/"
        self.desar_lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.npy"))]
        # self.desar_hr=[]
        # self.desar_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.npy"))]
        inputs_sy = self.opt['data_file_dir'] + "/denoising/UCMerced_LandUse/"
        desar_sy = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs_sy + "/*.tif"))]

        self.desar_counter = 0
        print("Total SAR despeckling training pairs : {}".format(len(self.desar_lr)))
        self.desar_lr = self.desar_lr * 10
        desar_sy = desar_sy * 2
        self.desar_lr += desar_sy
        random.shuffle(self.desar_lr)
        # self.desar_hr = self.desar_hr * 20
        print("Repeated SAR despeckling Dataset length : {}".format(len(self.desar_lr)))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.opt['gt_size'])
        ind_W = random.randint(0, W - self.opt['gt_size'])

        patch_1 = img_1[ind_H:ind_H + self.opt['gt_size'], ind_W:ind_W + self.opt['gt_size']]
        patch_2 = img_2[ind_H:ind_H + self.opt['gt_size'], ind_W:ind_W + self.opt['gt_size']]

        return patch_1, patch_2

    def _get_nonhazy_name(self, hazy_name):
        dir_name = os.path.dirname(os.path.dirname(hazy_name)) + "/clear"
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = os.path.splitext(hazy_name)[1]
        nonhazy_name = dir_name + "/" + name + suffix
        return nonhazy_name

    def _get_noncloud_name(self, cloud_name):
        dir_name = os.path.dirname(os.path.dirname(cloud_name)) + "/reference"
        name = cloud_name.split('/')[-1].split('.')[0]
        suffix = os.path.splitext(cloud_name)[1]
        noncloud_name = dir_name + "/" + name + suffix
        return noncloud_name

    def _get_nonsar_name(self, sar_name):
        dir_name = os.path.dirname(os.path.dirname(sar_name)) + "/model_1_org"
        name = sar_name.split('/')[-1].split('.')[0]
        suffix = os.path.splitext(sar_name)[1]
        nonsar_name = dir_name + "/" + name + ".adp.rmli" + suffix
        return nonsar_name

    def _get_nonshadow_name(self, shadow_name):
        dir_name = os.path.dirname(os.path.dirname(shadow_name)) + "/shadow_free"
        # name = shadow_name.split('/')[-1].split('.')[0]
        name = shadow_name.split('/')[-1][:-4]
        suffix = os.path.splitext(shadow_name)[1]
        nonshadow_name = dir_name + "/" + name + "_no_shadow" + suffix
        return nonshadow_name

class IRBenchmarks(data.Dataset):
    def __init__(self, args):
        super(IRBenchmarks, self).__init__()

        self.opt = args
        self.benchmarks = args['benchmarks']
        self.de_type = self.opt['de_type']
        self.de_dict = {dataset: idx for idx, dataset in enumerate(self.de_type)}

        self.toTensor = ToTensor()

        # self.resize = Resize(size=(512, 512), interpolation=InterpolationMode.NEAREST)
        self.resize = Resize(size=(128, 128), interpolation=InterpolationMode.NEAREST)

        self._init_lr()
        self.is_view_to_gray = False

    def __getitem__(self, idx):
        lr_sample = self.lr[idx]
        de_id = lr_sample["de_type"]
        clean_name = lr_sample["img"]

        if "denoise_15" in self.benchmarks or "denoise_25" in self.benchmarks or "denoise_50" in self.benchmarks or "denoise_100" in self.benchmarks or "denoise_75" in self.benchmarks:
            sigma = int(self.benchmarks[-1].split("_")[-1])
            hr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            lr, _ = self._add_gaussian_noise(hr, sigma)

            hr = np.concatenate((hr, hr, hr, hr, hr), axis=2)
            lr = np.concatenate((lr, lr, lr, lr, lr), axis=2)
        elif "deblurgauss_5" in self.benchmarks or "deblurgauss_10" in self.benchmarks:
            radius = int(self.benchmarks[-1].split("_")[-1])
            hr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            lr, _ = self._add_gaussian_blur(hr, radius)

            hr = np.concatenate((hr, hr, hr, hr, hr), axis=2)
            lr = np.concatenate((lr, lr, lr, lr, lr), axis=2)
        elif "decloud_rice1" in self.benchmarks:
            lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            clean_name = self._get_noncloud_name(lr_sample["img"])
            hr = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            hr = np.concatenate((hr, hr, hr, hr, hr), axis=2)
            lr = np.concatenate((lr, lr, lr, lr, lr), axis=2)
        elif "decloud_tcloud" in self.benchmarks:
            lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            clean_name = self._get_noncloud_name(lr_sample["img"])
            hr = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            hr = np.concatenate((hr, hr, hr, hr, hr), axis=2)
            lr = np.concatenate((lr, lr, lr, lr, lr), axis=2)

        elif "decloud_rice2" in self.benchmarks:
            lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            clean_name = self._get_noncloud_name(lr_sample["img"])
            hr = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            hr = np.concatenate((hr, hr, hr, hr, hr), axis=2)
            lr = np.concatenate((lr, lr, lr, lr, lr), axis=2)
        elif "deshadow" in self.benchmarks:
            lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            clean_name = self._get_nonshadow_name(lr_sample["img"])
            hr = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            hr = np.concatenate((hr, hr, hr, hr, hr), axis=2)
            lr = np.concatenate((lr, lr, lr, lr, lr), axis=2)
        elif "decloud_sen12ms" in self.benchmarks:
            lr, _, hr, clean_name = self.sen12dataset.getsen12ms_item(clean_name)  # lr（15,256,256）, hr 13,256,256
            lr = np.transpose(lr, (1, 2, 0))
            hr = np.transpose(hr, (1, 2, 0))
            lr = crop_img(lr, base=16)  # h,w,c
            hr = crop_img(hr, base=16)
            lr_sample["img"] = clean_name
        elif "desar" in self.benchmarks:
            # hr_sample = self.hr[idx]
            nrd_lq_path = lr_sample["img"]
            # nrd_lq_path = hr_sample["img"]
            nrd_gt_path = self._get_nonsar_name(lr_sample["img"])
            clean_name = nrd_lq_path
            nrd_ori_gt = np.load(nrd_gt_path)
            nrd_ori_lq = np.load(nrd_lq_path)  # H,W

            if len(nrd_ori_lq.shape) == 2:
                nrd_ori_lq = nrd_ori_lq[:, :, np.newaxis]
                nrd_ori_gt = nrd_ori_gt[:, :, np.newaxis]
            # nrd_ori_gt, nrd_ori_lq = paired_random_crop(nrd_ori_gt, nrd_ori_lq, self.crop_pad_size, 1,
            #                                             gt_path=nrd_gt_path)
            lr = crop_img(nrd_ori_lq, base=16)
            hr = crop_img(nrd_ori_gt, base=16)

            # intensity -> amplitude
            nrd_gt = np.sqrt(hr)
            nrd_lq = np.sqrt(lr)

            if self.is_view_to_gray:
                nrd_gt, _, _ = max_normalize(nrd_gt)  # amp to norm
                nrd_lq, _, _ = max_normalize(nrd_lq)
                nrd_gt = view_sar(nrd_gt)  # norm to gray
                nrd_lq = view_sar(nrd_lq)

            nrd_lq, nrd_lq_min, nrd_lq_max = max_normalize(nrd_lq)  # [0,1]
            nrd_gt_min, nrd_gt_max = nrd_lq_min, nrd_lq_max

            nrd_gt[np.isnan(nrd_gt)] = 0
            nrd_gt = np.abs((nrd_gt - nrd_lq_min) / (nrd_lq_max - nrd_lq_min))
            lr = nrd_lq
            hr = nrd_gt
            lr = np.repeat(lr, 3, axis=2)
            hr = np.repeat(hr, 3, axis=2)

            hr = np.concatenate((hr, hr, hr, hr, hr), axis=2)
            lr = np.concatenate((lr, lr, lr, lr, lr), axis=2)

            # HWC to CHW, numpy to tensor
            # nrd_lq = img2tensor(nrd_lq)
            # nrd_gt = img2tensor(nrd_gt)

        else:
            hr_sample = self.hr[idx]
            lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            hr = crop_img(np.array(Image.open(hr_sample["img"]).convert('RGB')), base=16)

        lr = self.toTensor(lr)
        hr = self.toTensor(hr)
        # lr = self.resize(lr)
        # hr = self.resize(hr)

        if "desar" in self.benchmarks:
            # return [lr_sample["img"], de_id], lr, hr, nrd_lq_min, nrd_lq_max
            return {'lq': lr, 'gt': hr, 'lq_path': lr_sample["img"], 'gt_path': clean_name, 'nrd_lq_min':nrd_lq_min, 'nrd_lq_max':nrd_lq_max}
        else:
            # return [lr_sample["img"], de_id], lr, hr
            return {'lq': lr, 'gt': hr, 'lq_path': lr_sample["img"], 'gt_path': clean_name}

    def __len__(self):
        return len(self.lr)

    def _init_lr(self):
        if 'denoise_15' in self.benchmarks:
            self._init_denoise(id=self.de_dict['denoise_15'])
        if 'denoise_25' in self.benchmarks:
            self._init_denoise(id=self.de_dict['denoise_25'])
        if 'denoise_50' in self.benchmarks:
            self._init_denoise(id=self.de_dict['denoise_50'])

        if 'decloud_rice1' in self.benchmarks:
            self._init_decloud_rice1(id=self.de_dict['decloud'])
        if 'decloud_rice2' in self.benchmarks:
            self._init_decloud_rice2(id=self.de_dict['decloud'])
        if 'decloud_tcloud' in self.benchmarks:
            self._init_decloud_tcloud(id=self.de_dict['decloud_tcloud'])
        if 'deshadow' in self.benchmarks:
            self._init_deshadow(self.de_dict['deshadow'])
        if 'desar' in self.benchmarks:
            self._init_desar(self.de_dict['desar'])
        if 'deblurgauss_5' in self.benchmarks:
            self._init_blurgauss(id=self.de_dict['deblurgauss_5'])
        if 'deblurgauss_10' in self.benchmarks:
            self._init_blurgauss(id=self.de_dict['deblurgauss_10'])
        if 'decloud_sen12ms' in self.benchmarks:
            self._init_decloud_sen12ms(self.de_dict['decloud_sen12ms'])

    def _get_nonsar_name(self, sar_name):
        dir_name = os.path.dirname(os.path.dirname(sar_name)) + "/model_1_org"
        name = sar_name.split('/')[-1].split('.')[0]
        suffix = os.path.splitext(sar_name)[1]
        nonsar_name = dir_name + "/" + name + ".adp.rmli" + suffix
        return nonsar_name

    def _get_noncloud_name(self, cloud_name):
        dir_name = os.path.dirname(os.path.dirname(cloud_name)) + "/reference"
        name = cloud_name.split('/')[-1].split('.')[0]
        suffix = os.path.splitext(cloud_name)[1]
        noncloud_name = dir_name + "/" + name + suffix
        return noncloud_name

    def _get_nonshadow_name(self, shadow_name):
        dir_name = os.path.dirname(os.path.dirname(shadow_name)) + "/shadow_free"
        # name = shadow_name.split('/')[-1].split('.')[0]
        name = shadow_name.split('/')[-1][:-4]
        suffix = os.path.splitext(shadow_name)[1]
        nonshadow_name = dir_name + "/" + name + "_free" + suffix
        return nonshadow_name

    def _add_gaussian_noise(self, clean_patch, sigma):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _add_gaussian_blur(self, clean_patch, radius=5):
        orig_patch = Image.fromarray(clean_patch)

        blurred_patch = orig_patch.filter(ImageFilter.GaussianBlur(radius))
        blurred_patch = np.array(blurred_patch)
        blurred_patch = np.clip(blurred_patch, 0, 255).astype(np.uint8)

        return blurred_patch, clean_patch


    ####################################################################################################
    ## DEBLURRING DATASET
    def _init_deblurring(self, benchmark, id):
        inputs = self.opt['data_file_dir'] + f"/deblurring/{benchmark}/test/input/"
        targets = self.opt['data_file_dir'] + f"/deblurring/{benchmark}/test/target/"

        self.lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.png"))]
        print("Total Deblur testing pairs : {}".format(len(self.hr)))

    ####################################################################################################
    ####################################################################################################
    ## DENOISING DATASET
    def _init_denoise(self, id):
        inputs = self.opt['data_file_dir'] + "/denoising/UCMerced_LandUse"

        clean = [x for x in sorted(glob.glob(inputs + "/*.tif"))]

        self.lr = [{"img": x, "de_type": id} for x in clean]
        self.hr = [{"img": x, "de_type": id} for x in clean]
        print("Total Denoise testing pairs : {}".format(len(self.lr)))

    def _init_blurgauss(self, id):
        inputs = self.opt['data_file_dir'] + "/deblur/hit-uav-DatasetNinja/img"

        clean = [x for x in sorted(glob.glob(inputs + f"/*.jpg"))]
        self.lr = [{"img": x, "de_type": id} for x in clean]
        self.hr = [{"img": x, "de_type": id} for x in clean]

        print("Total deblurgauss Ids testing : {}".format(len(self.lr)))

    def _init_decloud_rice1(self, id):
        """ Initialize the GoPro training dataset """
        inputs = self.opt['data_file_dir'] + "/decloud/rice1/cloud/"
        targets = self.opt['data_file_dir'] + "/decloud/rice1/reference/"
        decloud_rice1_lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.png"))]
        decloud_rice1_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.png"))]

        # inputs2 = self.args.data_file_dir + "/decloud/rice2/cloud/"
        # targets2 = self.args.data_file_dir + "/decloud/rice2/reference/"
        # decloud_rice2_lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs2 + "/*.png"))]
        # decloud_rice2_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets2 + "/*.png"))]

        self.lr = decloud_rice1_lr
        self.hr = decloud_rice1_hr
        print("Total Decloud test rice1 pairs : {}".format(len(self.lr)))

    def _init_decloud_rice2(self, id):
        """ Initialize the GoPro training dataset """
        # inputs = self.args.data_file_dir + "/decloud/rice1/cloud/"
        # targets = self.args.data_file_dir + "/decloud/rice1/reference/"
        # decloud_rice1_lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.png"))]
        # decloud_rice1_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.png"))]

        inputs2 = self.opt['data_file_dir'] + "/decloud/rice2/cloud/"
        targets2 = self.opt['data_file_dir'] + "/decloud/rice2/reference/"
        decloud_rice2_lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs2 + "/*.png"))]
        decloud_rice2_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets2 + "/*.png"))]

        self.lr = decloud_rice2_lr
        self.hr = decloud_rice2_hr
        print("Total Decloud test rice2 pairs : {}".format(len(self.lr)))

    def _init_decloud_tcloud(self, id):
        """ Initialize the GoPro training dataset """
        inputs = self.opt['data_file_dir'] + "/decloud/T-Cloud/cloud/"
        targets = self.opt['data_file_dir'] + "/decloud/T-Cloud/reference/"
        decloud_tcloud_lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.png"))]
        decloud_tcloud_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.png"))]

        self.lr = decloud_tcloud_lr
        self.hr = decloud_tcloud_hr
        print("Total Decloud Tcloud test pairs : {}".format(len(self.lr)))

    def _init_decloud_sen12ms(self, id):
        """ Initialize the sen12ms training dataset """
        train_datset_dir = self.opt['data_file_dir'] + "/decloud/SEN12MS-CR"
        self.sen12dataset = SEN12MSCRDataset(train_datset_dir)

        decloud_sen12ms_lr = [{"img": x, "de_type": id} for x in self.sen12dataset.img]

        self.lr = decloud_sen12ms_lr
        self.hr = decloud_sen12ms_lr

        print("Total Decloud sen12ms test pairs : {}".format(len(self.lr)))
        # print("Repeated Decloud sen12ms Dataset length : {}".format(len(self.decloud_sen12ms_lr)))

    def _init_deshadow(self, id):
        """ Initialize the GoPro training dataset """
        inputs = self.opt['data_file_dir'] + "/deshadow/SRD/shadow/"
        targets = self.opt['data_file_dir'] + "/deshadow/SRD/shadow_free/"
        self.lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.jpg"))]
        self.hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.jpg"))]

        print("Total Deshadow test pairs : {}".format(len(self.hr)))

    def _init_desar(self, id):
        """ Initialize the GoPro training dataset """
        inputs = self.opt['data_file_dir'] + "/despeckling/SARdata-512/ori/"
        targets = self.opt['data_file_dir'] + "/despeckling/SARdata-512/model_1_org/"
        self.lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.npy"))]
        self.hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.npy"))]

        print("Total SAR despeckling test pairs : {}".format(len(self.hr)))

class IRBenchmarks_test(data.Dataset):
    def __init__(self, args):
        super(IRBenchmarks_test, self).__init__()

        self.opt = args
        self.benchmarks = args.benchmarks
        self.de_type = self.opt.de_type
        self.de_dict = {dataset: idx for idx, dataset in enumerate(self.de_type)}

        self.toTensor = ToTensor()

        self.resize = Resize(size=(512, 512), interpolation=InterpolationMode.NEAREST)
        # self.resize = Resize(size=(128, 128), interpolation=InterpolationMode.NEAREST)

        self._init_lr()
        self.is_view_to_gray = False

    def __getitem__(self, idx):
        lr_sample = self.lr[idx]
        de_id = lr_sample["de_type"]
        clean_name = lr_sample["img"]

        if "denoise_15" in self.benchmarks or "denoise_25" in self.benchmarks or "denoise_50" in self.benchmarks or "denoise_100" in self.benchmarks or "denoise_75" in self.benchmarks:
            sigma = int(self.benchmarks[-1].split("_")[-1])
            hr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            lr, _ = self._add_gaussian_noise(hr, sigma)

            hr = np.concatenate((hr, hr, hr, hr, hr), axis=2)
            lr = np.concatenate((lr, lr, lr, lr, lr), axis=2)
        elif "deblurgauss_5" in self.benchmarks or "deblurgauss_10" in self.benchmarks:
            radius = int(self.benchmarks[-1].split("_")[-1])
            hr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            lr, _ = self._add_gaussian_blur(hr, radius)

            hr = np.concatenate((hr, hr, hr, hr, hr), axis=2)
            lr = np.concatenate((lr, lr, lr, lr, lr), axis=2)
        elif "decloud_rice1" in self.benchmarks:
            lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            clean_name = self._get_noncloud_name(lr_sample["img"])
            hr = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            hr = np.concatenate((hr, hr, hr, hr, hr), axis=2)
            lr = np.concatenate((lr, lr, lr, lr, lr), axis=2)
        elif "decloud_tcloud" in self.benchmarks:
            lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            clean_name = self._get_noncloud_name(lr_sample["img"])
            hr = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            hr = np.concatenate((hr, hr, hr, hr, hr), axis=2)
            lr = np.concatenate((lr, lr, lr, lr, lr), axis=2)

        elif "decloud_rice2" in self.benchmarks:
            lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            clean_name = self._get_noncloud_name(lr_sample["img"])
            hr = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            hr = np.concatenate((hr, hr, hr, hr, hr), axis=2)
            lr = np.concatenate((lr, lr, lr, lr, lr), axis=2)
        elif "deshadow" in self.benchmarks:
            lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            clean_name = self._get_nonshadow_name(lr_sample["img"])
            hr = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            hr = np.concatenate((hr, hr, hr, hr, hr), axis=2)
            lr = np.concatenate((lr, lr, lr, lr, lr), axis=2)

        elif "decloud_sen12ms" in self.benchmarks:
            lr, _, hr, clean_name = self.sen12dataset.getsen12ms_item(clean_name)  # lr（15,256,256）, hr 13,256,256
            lr = np.transpose(lr, (1, 2, 0))
            hr = np.transpose(hr, (1, 2, 0))
            lr = crop_img(lr, base=16)  # h,w,c
            hr = crop_img(hr, base=16)
            lr_sample["img"] = clean_name

        elif "desar" in self.benchmarks:
            # hr_sample = self.hr[idx]
            nrd_lq_path = lr_sample["img"]
            # nrd_lq_path = hr_sample["img"]
            nrd_gt_path = self._get_nonsar_name(lr_sample["img"])
            clean_name = nrd_lq_path
            nrd_ori_gt = np.load(nrd_gt_path)
            nrd_ori_lq = np.load(nrd_lq_path)  # H,W

            if len(nrd_ori_lq.shape) == 2:
                nrd_ori_lq = nrd_ori_lq[:, :, np.newaxis]
                nrd_ori_gt = nrd_ori_gt[:, :, np.newaxis]
            # nrd_ori_gt, nrd_ori_lq = paired_random_crop(nrd_ori_gt, nrd_ori_lq, self.crop_pad_size, 1,
            #                                             gt_path=nrd_gt_path)
            lr = crop_img(nrd_ori_lq, base=16)
            hr = crop_img(nrd_ori_gt, base=16)

            # intensity -> amplitude
            nrd_gt = np.sqrt(hr)
            nrd_lq = np.sqrt(lr)

            if self.is_view_to_gray:
                nrd_gt, _, _ = max_normalize(nrd_gt)  # amp to norm
                nrd_lq, _, _ = max_normalize(nrd_lq)
                nrd_gt = view_sar(nrd_gt)  # norm to gray
                nrd_lq = view_sar(nrd_lq)

            nrd_lq, nrd_lq_min, nrd_lq_max = max_normalize(nrd_lq)  # [0,1]
            nrd_gt_min, nrd_gt_max = nrd_lq_min, nrd_lq_max

            nrd_gt[np.isnan(nrd_gt)] = 0
            nrd_gt = np.abs((nrd_gt - nrd_lq_min) / (nrd_lq_max - nrd_lq_min))
            lr = nrd_lq
            hr = nrd_gt
            lr = np.repeat(lr, 3, axis=2)
            hr = np.repeat(hr, 3, axis=2)

            hr = np.concatenate((hr, hr, hr, hr, hr), axis=2)
            lr = np.concatenate((lr, lr, lr, lr, lr), axis=2)

            # HWC to CHW, numpy to tensor
            # nrd_lq = img2tensor(nrd_lq)
            # nrd_gt = img2tensor(nrd_gt)

        else:
            hr_sample = self.hr[idx]
            lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            hr = crop_img(np.array(Image.open(hr_sample["img"]).convert('RGB')), base=16)

        lr = self.toTensor(lr)
        hr = self.toTensor(hr)
        # lr = self.resize(lr)
        # hr = self.resize(hr)

        if "desar" in self.benchmarks:
            # return [lr_sample["img"], de_id], lr, hr, nrd_lq_min, nrd_lq_max
            return {'lq': lr, 'gt': hr, 'lq_path': lr_sample["img"], 'gt_path': clean_name, 'nrd_lq_min':nrd_lq_min, 'nrd_lq_max':nrd_lq_max}
        else:
            # return [lr_sample["img"], de_id], lr, hr
            return {'lq': lr, 'gt': hr, 'lq_path': lr_sample["img"], 'gt_path': clean_name}

    def __len__(self):
        return len(self.lr)

    def _init_lr(self):
        if 'denoise_15' in self.benchmarks:
            self._init_denoise(id=self.de_dict['denoise_15'])
        if 'denoise_25' in self.benchmarks:
            self._init_denoise(id=self.de_dict['denoise_15'])
        if 'denoise_50' in self.benchmarks:
            self._init_denoise(id=self.de_dict['denoise_15'])

        if 'decloud_rice1' in self.benchmarks:
            self._init_decloud_rice1(id=self.de_dict['decloud'])
        if 'decloud_rice2' in self.benchmarks:
            self._init_decloud_rice2(id=self.de_dict['decloud'])
        if 'decloud_tcloud' in self.benchmarks:
            self._init_decloud_tcloud(id=self.de_dict['decloud'])
        if 'deshadow' in self.benchmarks:
            self._init_deshadow(self.de_dict['deshadow'])
        if 'desar' in self.benchmarks:
            self._init_desar(self.de_dict['desar'])
        if 'deblurgauss_5' in self.benchmarks:
            self._init_blurgauss(id=self.de_dict['deblurgauss_5'])
        if 'deblurgauss_10' in self.benchmarks:
            self._init_blurgauss(id=self.de_dict['deblurgauss_10'])
        if 'decloud_sen12ms' in self.benchmarks:
            self._init_decloud_sen12ms(self.de_dict['decloud_sen12ms'])

    def _get_nonsar_name(self, sar_name):
        dir_name = os.path.dirname(os.path.dirname(sar_name)) + "/model_1_org"
        name = sar_name.split('/')[-1].split('.')[0]
        suffix = os.path.splitext(sar_name)[1]
        nonsar_name = dir_name + "/" + name + ".adp.rmli" + suffix
        return nonsar_name

    def _get_noncloud_name(self, cloud_name):
        dir_name = os.path.dirname(os.path.dirname(cloud_name)) + "/reference"
        name = cloud_name.split('/')[-1].split('.')[0]
        suffix = os.path.splitext(cloud_name)[1]
        noncloud_name = dir_name + "/" + name + suffix
        return noncloud_name

    def _get_nonshadow_name(self, shadow_name):
        dir_name = os.path.dirname(os.path.dirname(shadow_name)) + "/shadow_free"
        # name = shadow_name.split('/')[-1].split('.')[0]
        name = shadow_name.split('/')[-1][:-4]
        suffix = os.path.splitext(shadow_name)[1]
        nonshadow_name = dir_name + "/" + name + "_free" + suffix
        return nonshadow_name

    def _add_gaussian_noise(self, clean_patch, sigma):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _add_gaussian_blur(self, clean_patch, radius=5):
        orig_patch = Image.fromarray(clean_patch)

        blurred_patch = orig_patch.filter(ImageFilter.GaussianBlur(radius))
        blurred_patch = np.array(blurred_patch)
        blurred_patch = np.clip(blurred_patch, 0, 255).astype(np.uint8)

        return blurred_patch, clean_patch

    ####################################################################################################
    ## DEBLURRING DATASET
    def _init_deblurring(self, benchmark, id):
        inputs = self.opt['data_file_dir'] + f"/deblurring/{benchmark}/test/input/"
        targets = self.opt['data_file_dir'] + f"/deblurring/{benchmark}/test/target/"

        self.lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.png"))]
        print("Total Deblur testing pairs : {}".format(len(self.hr)))

    ####################################################################################################
    ####################################################################################################
    ## DENOISING DATASET
    def _init_denoise(self, id):
        inputs = self.opt.data_file_dir + "/denoising/UCMerced_LandUse"

        clean = [x for x in sorted(glob.glob(inputs + "/*.tif"))]

        self.lr = [{"img": x, "de_type": id} for x in clean]
        self.hr = [{"img": x, "de_type": id} for x in clean]
        print("Total Denoise testing pairs : {}".format(len(self.lr)))

    def _init_blurgauss(self, id):
        inputs = self.opt.data_file_dir + "/deblur/hit-uav-DatasetNinja/img"

        clean = [x for x in sorted(glob.glob(inputs + f"/*.jpg"))]
        self.lr = [{"img": x, "de_type": id} for x in clean]
        self.hr = [{"img": x, "de_type": id} for x in clean]

        print("Total deblurgauss Ids testing : {}".format(len(self.lr)))


    def _init_decloud_rice1(self, id):
        """ Initialize the GoPro training dataset """
        inputs = self.opt.data_file_dir + "/decloud/rice1/cloud/"
        targets = self.opt.data_file_dir + "/decloud/rice1/reference/"
        decloud_rice1_lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.png"))]
        decloud_rice1_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.png"))]

        # inputs2 = self.args.data_file_dir + "/decloud/rice2/cloud/"
        # targets2 = self.args.data_file_dir + "/decloud/rice2/reference/"
        # decloud_rice2_lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs2 + "/*.png"))]
        # decloud_rice2_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets2 + "/*.png"))]

        self.lr = decloud_rice1_lr
        self.hr = decloud_rice1_hr
        print("Total Decloud test rice1 pairs : {}".format(len(self.lr)))

    def _init_decloud_rice2(self, id):
        """ Initialize the GoPro training dataset """
        # inputs = self.args.data_file_dir + "/decloud/rice1/cloud/"
        # targets = self.args.data_file_dir + "/decloud/rice1/reference/"
        # decloud_rice1_lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.png"))]
        # decloud_rice1_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.png"))]

        inputs2 = self.opt.data_file_dir + "/decloud/rice2/cloud/"
        targets2 = self.opt.data_file_dir + "/decloud/rice2/reference/"
        decloud_rice2_lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs2 + "/*.png"))]
        decloud_rice2_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets2 + "/*.png"))]

        self.lr = decloud_rice2_lr
        self.hr = decloud_rice2_hr
        print("Total Decloud test rice2 pairs : {}".format(len(self.lr)))

    def _init_decloud_tcloud(self, id):
        """ Initialize the GoPro training dataset """
        inputs = self.opt.data_file_dir + "/decloud/T-Cloud/cloud/"
        targets = self.opt.data_file_dir + "/decloud/T-Cloud/reference/"
        decloud_tcloud_lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.png"))]
        decloud_tcloud_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.png"))]

        self.lr = decloud_tcloud_lr
        self.hr = decloud_tcloud_hr
        print("Total Decloud Tcloud test pairs : {}".format(len(self.lr)))

    def _init_decloud_sen12ms(self, id):
        """ Initialize the sen12ms training dataset """
        train_datset_dir = self.opt.data_file_dir + "/decloud/SEN12MS-CR/"
        print(train_datset_dir)
        self.sen12dataset = SEN12MSCRDataset(train_datset_dir)

        decloud_sen12ms_lr = [{"img": x, "de_type": id} for x in self.sen12dataset.img]

        self.lr = decloud_sen12ms_lr
        self.hr = decloud_sen12ms_lr
        print("Total Decloud sen12ms test pairs : {}".format(len(self.lr)))

    def _init_deshadow(self, id):
        """ Initialize the GoPro training dataset """
        inputs = self.opt.data_file_dir + "/deshadow/SRD/shadow/"
        targets = self.opt.data_file_dir + "/deshadow/SRD/shadow_free/"
        self.lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.jpg"))]
        self.hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.jpg"))]

        print("Total Deshadow test pairs : {}".format(len(self.hr)))

    def _init_desar(self, id):
        """ Initialize the GoPro training dataset """
        inputs = self.opt.data_file_dir + "/despeckling/SARdata-512/ori/"
        targets = self.opt.data_file_dir + "/despeckling/SARdata-512/model_1_org/"
        self.lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.npy"))]
        self.hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.npy"))]

        print("Total SAR despeckling test pairs : {}".format(len(self.hr)))

