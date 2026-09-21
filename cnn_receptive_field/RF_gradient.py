import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(0) # 재현성

net = nn.Sequential(
    nn.Conv2d(1, 4, kernel_size=3, padding=0),  # 1 -> 4채널
    nn.ReLU(),
    nn.Conv2d( 4, 4, kernel_size=3, padding=0), # 4 -> 4
    nn. ReLU(),
    nn.Conv2d(4, 1, kernel_size=3, padding=0),  # 4 -> 1
)
net.eval()

x = torch.zeros(1, 1, 28, 28, requires_grad=True) # 값이 0이어도 gradient는 흐름
y = net(x)
print('출력 feature map 크기:', tuple(y.shape)) # 행렬 y의 차원 출력

oy, ox = 10, 10     # 출력에서 볼 위치
target = y[0, 0, oy, ox]   # 볼 위치의 한 점 스칼라
net.zero_grad()
target.backward()  # 입력까지 gradient 전파

grad = x.grad[0, 0].abs().numpy()
rf_mask = (grad > 0).astype(float)
ys, xs = np.where(rf_mask > 0)
print(f'이 뉴런의 receptive field 크기: {ys.max()-ys.min()+1} x {xs.max()-xs.min()+1} (기댓값 7x7)')

import matplotlib.pyplot as plt
plt.figure(figsize=(4,4))
plt.title('RF of output neuron at (10, 10): 7x7') ## 22x22 최종 feature map에서의 (10,10) 뉴런에서의 출력
# 최종 feature map의 (10, 10)에서의 출력은 
# 원본 이미지에서 row: 10~16, col: 10~16을 보게 된다. (총 7x7 RF)
plt.imshow(rf_mask, cmap='gray')
plt.savefig('receptive_field.png', dpi=120)
print('저장 완료: receptive_field.png')

plt.show()
# 출력 결과에서 흰색으로 표시된 영역이 RF 7x7 영역인 것이다.