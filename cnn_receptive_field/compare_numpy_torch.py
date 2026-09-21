import numpy as np
import torch 
import torch.nn.functional as F

'''
nunmpy로 conv2d 구현
'''

def conv2d_numpy(image, kernel, stride=1, padding=0):
    kh, kw = kernel.shape
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)
        
    H, W = image.shape
    Ho = (H - kh) // stride + 1
    Wo = (W - kw) // stride + 1
    out = np.zeros((Ho, Wo), dtype=float)
    
    for i in range(Ho):
        for j in range(Wo):
            r, c = i * stride, j * stride
            patch = image[r:r+kh, c:c+kw]
            out[i, j] = np.sum(patch * kernel)
    return out


edge_kernel = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]], dtype=float)

img = np.zeros((8,8), dtype=float)
img[:, 4:] = 1.0

out_np = conv2d_numpy(img, edge_kernel, stride=1, padding=0)
print('입력 영상 (0/1):')
print(img.astype(int))
print('\n numpy conv 출력 (세로 경계에서 반응):')
print(np.round(out_np, 1))
print('\n 최대 |반응| 위치의 값:', out_np[np.unravel_index(np.argmax(np.abs(out_np)), out_np.shape)])



'''
torch로 conv2d 구현
'''
img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
ker_t = torch.from_numpy(edge_kernel).float().unsqueeze(0).unsqueeze(0)

out_torch = F.conv2d(img_t, ker_t, stride=1, padding=0)[0, 0].numpy()
print('torch conv 출력:')
print(np.round(out_torch, 1))

same = np.allclose(out_np, out_torch)
print('\n numpy 결과와 torch 결과가 동일한가? ->', same)
print('최대 절대 오차:', np.max(np.abs(out_np - out_torch)))
