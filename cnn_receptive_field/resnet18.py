import torch
import matplotlib.pyplot as plt

from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # ImageNet 학습 가중치
trained = True

model.eval()
w = model.conv1.weight.data # [64, 3, 7, 7]
print('conv1 가중치 모양:', tuple(w.shape))

w_min, w_max = w.min(), w.max()
w_norm = (w - w_min) / (w_max - w_min + 1e-8)  # 0~1 스케일

fig, axes = plt.subplots(8, 8, figsize=(8, 8))
fig.suptitle('ResNet18 conv1 filters ' + ('(trained)' if trained else '(random)'))

for i, ax in enumerate(axes.flat):
    kernel = w_norm[i].permute(1, 2, 0).numpy()  # [3,7,7] -> [7,7,3]
    ax.imshow(kernel)
    ax.axis('off')
plt.tight_layout()
plt.savefig('conv1_filters.png', dpi=120)
print('저장 완료: conv1_filters.png')
# plt.show()

