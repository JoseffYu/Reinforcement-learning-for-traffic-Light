import torch
import torch.nn as nn
import numpy as np

arr = [[[[ 0.0413361 ,  0.01061872, -0.02137395, -0.00487474]],{}]]  # 一维数组

# 将一维数组变成二维数组（在第二个轴上增加新的维度）
expanded_arr = np.expand_dims(arr,0)

print(arr)
print(expanded_arr)