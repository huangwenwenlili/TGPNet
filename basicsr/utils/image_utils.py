"""
Created on 2020/9/8

@author: Boyun Li
"""
import os
import cv2
import numpy as np
import torch
import random
import torch.nn as nn
from torch.nn import init
from PIL import Image
from torchvision.utils import make_grid
import math

class EdgeComputation(nn.Module):
    def __init__(self, test=False):
        super(EdgeComputation, self).__init__()
        self.test = test
    def forward(self, x):
        if self.test:
            x_diffx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
            x_diffy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

            # y = torch.Tensor(x.size()).cuda()
            y = torch.Tensor(x.size())
            y.fill_(0)
            y[:, :, :, 1:] += x_diffx
            y[:, :, :, :-1] += x_diffx
            y[:, :, 1:, :] += x_diffy
            y[:, :, :-1, :] += x_diffy
            y = torch.sum(y, 1, keepdim=True) / 3
            y /= 4
            return y
        else:
            x_diffx = torch.abs(x[:, :, 1:] - x[:, :, :-1])
            x_diffy = torch.abs(x[:, 1:, :] - x[:, :-1, :])

            y = torch.Tensor(x.size())
            y.fill_(0)
            y[:, :, 1:] += x_diffx
            y[:, :, :-1] += x_diffx
            y[:, 1:, :] += x_diffy
            y[:, :-1, :] += x_diffy
            y = torch.sum(y, 0) / 3
            y /= 4
            return y.unsqueeze(0)

from skimage.transform import resize
# randomly crop a patch from image
def crop_patch(im, pch_size):
    H = im.shape[0]
    W = im.shape[1]
    ind_H = random.randint(0, H - pch_size)
    ind_W = random.randint(0, W - pch_size)
    pch = im[ind_H:ind_H + pch_size, ind_W:ind_W + pch_size]
    return pch


# crop an image to the multiple of base
def crop_img(image, base=64,min_width=128):
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    img = image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]
    if img.shape[0] < min_width or img.shape[1] < min_width:
        img = resize(img, (min_width, min_width, image.shape[2]), order=1).astype(np.uint8)  # order=1 for bilinear interpolation
    return img

# image (H, W, C) -> patches (B, H, W, C)
def slice_image2patches(image, patch_size=64, overlap=0):
    assert image.shape[0] % patch_size == 0 and image.shape[1] % patch_size == 0
    H = image.shape[0]
    W = image.shape[1]
    patches = []
    image_padding = np.pad(image, ((overlap, overlap), (overlap, overlap), (0, 0)), mode='edge')
    for h in range(H // patch_size):
        for w in range(W // patch_size):
            idx_h = [h * patch_size, (h + 1) * patch_size + overlap]
            idx_w = [w * patch_size, (w + 1) * patch_size + overlap]
            patches.append(np.expand_dims(image_padding[idx_h[0]:idx_h[1], idx_w[0]:idx_w[1], :], axis=0))
    return np.concatenate(patches, axis=0)


# patches (B, H, W, C) -> image (H, W, C)
def splice_patches2image(patches, image_size, overlap=0):
    assert len(image_size) > 1
    assert patches.shape[-3] == patches.shape[-2]
    H = image_size[0]
    W = image_size[1]
    patch_size = patches.shape[-2] - overlap
    image = np.zeros(image_size)
    idx = 0
    for h in range(H // patch_size):
        for w in range(W // patch_size):
            image[h * patch_size:(h + 1) * patch_size, w * patch_size:(w + 1) * patch_size, :] = patches[idx,
                                                                                                 overlap:patch_size + overlap,
                                                                                                 overlap:patch_size + overlap,
                                                                                                 :]
            idx += 1
    return image


# def data_augmentation(image, mode):
#     if mode == 0:
#         # original
#         out = image.numpy()
#     elif mode == 1:
#         # flip up and down
#         out = np.flipud(image)
#     elif mode == 2:
#         # rotate counterwise 90 degree
#         out = np.rot90(image, axes=(1, 2))
#     elif mode == 3:
#         # rotate 90 degree and flip up and down
#         out = np.rot90(image, axes=(1, 2))
#         out = np.flipud(out)
#     elif mode == 4:
#         # rotate 180 degree
#         out = np.rot90(image, k=2, axes=(1, 2))
#     elif mode == 5:
#         # rotate 180 degree and flip
#         out = np.rot90(image, k=2, axes=(1, 2))
#         out = np.flipud(out)
#     elif mode == 6:
#         # rotate 270 degree
#         out = np.rot90(image, k=3, axes=(1, 2))
#     elif mode == 7:
#         # rotate 270 degree and flip
#         out = np.rot90(image, k=3, axes=(1, 2))
#         out = np.flipud(out)
#     else:
#         raise Exception('Invalid choice of image transformation')
#     return out

def data_augmentation(image, mode):
    if mode == 0:
        # original
        out = image.numpy()
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')
    return out


# def random_augmentation(*args):
#     out = []
#     if random.randint(0, 1) == 1:
#         flag_aug = random.randint(1, 7)
#         for data in args:
#             out.append(data_augmentation(data, flag_aug).copy())
#     else:
#         for data in args:
#             out.append(data)
#     return out

def random_augmentation(*args):
    out = []
    flag_aug = random.randint(1, 7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out


def weights_init_normal_(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.apply(weights_init_normal_)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()
    # return img_var.detach().cpu().numpy()[0]


def save_image(name, image_np, output_path="output/normal/"):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    p = np_to_pil(image_np)
    p.save(output_path + "{}.png".format(name))


def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

import cv2
def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED
    }
    if img_np is None:
        raise Exception('None .. !!!')
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img

def max_normalize(img, img_max=0):
    img = np.abs(img)
    img[np.isnan(img)] = 1e-9
    img_min = np.min(img[img != 0])
    if not img_max:
        img_max = np.max(img)
    img = np.abs((img - img_min) / (img_max - img_min))
    return img, img_min, img_max


def max_denormalize(img, img_min, img_max):
    img = img * (img_max - img_min) + img_min
    return img

def normalizedAmp2intensity(img, img_min, img_max):
    img = max_denormalize(img, img_min, img_max)
    img = np.power(img, 2)
    return img

def intensity2normalizedAmp(img):
    img = np.power(img, 0.5)
    img, _, _ = max_normalize(img)
    return img

def sar_val_normalize(img, img_min, img_max):
    img = normalizedAmp2intensity(img, img_min, img_max)
    img = intensity2normalizedAmp(img)
    return img


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)

# SAR Data Augmentation

def add_noise(image, L=None):
    if L is None:
        L = int(np.random.choice([1, 2, 3, 4]))
    assert isinstance(L, int)
    img_max = np.max(image)

    img_size_numpy = image.shape
    rows = img_size_numpy[0]
    columns = img_size_numpy[1]
    s = np.zeros((1, rows, columns))
    for k in range(0, L):
        gamma = np.abs(np.random.randn(1, rows, columns) + np.random.randn(1, rows, columns) * 1j) ** 2 / 2
        s = s + gamma
    s_amplitude = np.sqrt(s / L).squeeze(0)
    if len(image.shape) >= 3:
        s_amplitude = s_amplitude[:, :, np.newaxis]

    noisy_image = np.multiply(image, s_amplitude)
    # mean = 0
    # img_p5 = np.percentile(image, 50)
    # std = noise_level * img_p5
    # noise = np.random.normal(mean, std, image.shape)
    # noisy_image = image + noise
    return np.clip(noisy_image, 0, img_max)


def adjust_contrast(lq, gt):
    mean = np.mean(gt)
    img_max = np.max(gt)
    factor = np.clip(random.random() * 2, 0.5, 1.5)
    lq, gt = map(lambda img: np.clip((img - mean) * factor + mean, 0, img_max), [lq, gt])
    return lq, gt


def adjust_brightness(lq, gt):
    img_p5 = np.percentile(gt, 50)
    img_max = np.max(gt)
    factor = np.clip(random.random() * 2 - 1, -0.5, 0.5)
    lq, gt = map(lambda img: np.clip(img + factor * img_p5, 0, img_max), [lq, gt])

    return lq, gt


def adjust_gamma(lq, gt):
    img_max = np.max(gt)
    gamma = max(0.9, random.random())
    lq, gt = map(lambda img: np.clip(np.power(img, gamma), 0, img_max), [lq, gt])
    return lq, gt


def generate_displacement_field(image_shape, alpha, sigma):
    random_field_x = np.random.uniform(-1, 1, image_shape) * alpha
    random_field_y = np.random.uniform(-1, 1, image_shape) * alpha

    # 高斯滤波平滑位移场
    smooth_field_x = gaussian_filter(random_field_x, [sigma, sigma])
    smooth_field_y = gaussian_filter(random_field_y, [sigma, sigma])

    return smooth_field_x, smooth_field_y


from scipy.ndimage import map_coordinates


def elastic_transform(image, displacement_field, interpolation_order=1):
    dx, dy = displacement_field

    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    transformed_image = map_coordinates(image, indices, order=interpolation_order, mode='reflect').reshape(image.shape)

    return transformed_image


def adjust_elasticity(lq, gt):
    alpha = 30 + int(random.random() * 30)
    # sigma = np.random.randn(1) * 4
    displacement_field = generate_displacement_field(gt.shape, alpha, 4)

    lq = elastic_transform(lq, displacement_field)
    gt = elastic_transform(gt, displacement_field)
    return lq, gt


def sar_augment(lq, gt, speckle=False, contrast=False, brightness=False, gamma=False, elastic=False):
    speckle = speckle and random.random() < 0.5
    contrast = contrast and random.random() < 0.5
    brightness = not contrast and brightness and random.random() < 0.5
    gamma = gamma and random.random() < 0.5
    elastic = elastic and random.random() < 0.5

    if speckle:
        lq = add_noise(lq)

    if contrast:
        lq, gt = adjust_contrast(lq, gt)

    if brightness:
        lq, gt = adjust_brightness(lq, gt)

    if gamma:
        lq, gt = adjust_gamma(lq, gt)

    if elastic:
        lq, gt = adjust_elasticity(lq, gt)

    return lq, gt


def view_sar(img, ratio=0.3):
    _sum = np.sum(img)
    _len = len(np.nonzero(img)[0])
    scale = ratio / (_sum / _len)
    img = img * scale
    img = np.where(img > 1, 1, img)
    img = (img * 255.)
    return img

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def splitimage(imgtensor, crop_size=128, overlap_size=64):
    _, C, H, W = imgtensor.shape
    hstarts = [x for x in range(0, H, crop_size - overlap_size)]
    while hstarts and hstarts[-1] + crop_size >= H:
        hstarts.pop()
    hstarts.append(H - crop_size)
    wstarts = [x for x in range(0, W, crop_size - overlap_size)]
    while wstarts and wstarts[-1] + crop_size >= W:
        wstarts.pop()
    wstarts.append(W - crop_size)
    starts = []
    split_data = []
    for hs in hstarts:
        for ws in wstarts:
            cimgdata = imgtensor[:, :, hs:hs + crop_size, ws:ws + crop_size]
            starts.append((hs, ws))
            split_data.append(cimgdata)
    return split_data, starts

def get_scoremap(H, W, C, B=1, is_mean=True):
    center_h = H / 2
    center_w = W / 2

    score = torch.ones((B, C, H, W))
    if not is_mean:
        for h in range(H):
            for w in range(W):
                score[:, :, h, w] = 1.0 / (math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-6))
    return score

def mergeimage(split_data, starts, crop_size = 128, resolution=(1, 3, 128, 128)):
    B, C, H, W = resolution[0], resolution[1], resolution[2], resolution[3]
    tot_score = torch.zeros((B, C, H, W))
    merge_img = torch.zeros((B, C, H, W))
    scoremap = get_scoremap(crop_size, crop_size, C, B=B, is_mean=True)
    for simg, cstart in zip(split_data, starts):
        hs, ws = cstart
        merge_img[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap * simg
        tot_score[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap
    merge_img = merge_img / tot_score
    return merge_img