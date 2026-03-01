import torch
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import np_metric as img_met

from basicsr.models.archs.TGP_Net import TGPNet

from fvcore.nn import FlopCountAnalysis, flop_count_table


# Get model weights and parameters
import yaml

yaml_file = './option/TGPNet_test.yml'

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')

model = TGPNet(**x['network_g'])


g = model.cuda()
g.eval()
a = torch.randn(1,15,256,256).cuda()
type_npl = np.load('./data/type7.npy').astype(np.float32)
typep = torch.from_numpy(type_npl).clone()
img_typee = typep[0]

with torch.no_grad():
    flops = FlopCountAnalysis(g, inputs=[a, torch.unsqueeze(img_typee.cuda(), 0)])
    print(flop_count_table(flops))

# 打印自定义格式化的 FLOPs 表格
# print(custom_flop_count_table(flops, decimal_places=2))

# random_input = torch.randn(1, 3, 512, 512).cuda()
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
iterations = 100   # 重复计算的轮次
# GPU预热
g.eval()
with torch.no_grad():
    for i in range(50):
        _ = g([a, torch.unsqueeze(img_typee.cuda(), 0)])

# 测速
times = torch.zeros(iterations)     # 存储每轮iteration的时间
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = g([a, torch.unsqueeze(img_typee.cuda(), 0)])
        ender.record()
        # 同步GPU时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) # 计算时间
        times[iter] = curr_time
        # print(curr_time)

mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))

