import torch
import torch.nn.functional as F
import numpy as np

edge_kernel = np.array([
    [1,0,-1],
    [1,0,-1],
    [1,0,-1]
], dtype=float)

img = np.zeros((8,8), dtype=float) # 8x8 이미지
img[:, 4:] = 1.0 # 모든 행의 4번 열부터 끝까지 1.0으로 채운다.

# numpy 배열을 torch 텐서로: conv2d는 (N, C, H, W) / (out_c, in_c, kh, kw) 모양 요구
img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)  #(1, 1, 8, 8)
ker_t = torch.from_numpy(edge_kernel).float().unsqueeze(0).unsqueeze(0) # (1, 1, 3, 3)

out_torch = F.conv2d(img_t, ker_t, stride=1, padding=0)[0, 0].numpy()
print('torch conv 출력')
print(np.round(out_torch, 1))
