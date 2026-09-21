import numpy as np

# 함수를 먼저 정의
def conv2d_numpy(image, kernel, stride=1, padding=0):
    """numpy로 처음부터 구현한 2D 컨볼루션(정확히는 cross-correlation, 딥러닝 conv와 동일).
    image: (H, W), kernel: (kh, kw) -> 출력 (Ho, Wo)
    입력크기: 8x8
    커널크기: 3x3
    stride = 1
    padding = 0
    """
    kh, kw = kernel.shape   # 커널 크기 구하기 (kh=3, kw=3)
    if padding > 0:        # padding=0이므로 실행 X
        image = np.pad(image, padding, mode='constant', constant_values=0)

    H, W = image.shape # 입력 이미지 크기 대입
    Ho = (H - kh) // stride + 1     # 출력 Height = (입력 - 커널 + 2*패딩)/스트라이드 + 1 = 6
    Wo = (W - kw) // stride + 1     # 출력 Width
    out = np.zeros((Ho, Wo), dtype=float)   # 출력 배열을 미리 생성(6x6을 0으로 채움). 여기다 conv 결과를 하나씩 채워 넣음.
    for i in range(Ho):            # 출력 픽셀마다 (i=0~5, j=0~5)
        for j in range(Wo):
            r, c = i * stride, j * stride   # 커널의 실제 위치 계산
            patch = image[r:r+kh, c:c+kw]   # 커널이 덮는 이웃을 가져온다. 
            out[i, j] = np.sum(patch * kernel)  # 실제 conv 연산(커널곱을 전부 합함)
    return out


vertical = np.array([[1,0,-1], [1,0,-1], [1,0,-1]], dtype=float) #세로 엣지
horizontal = np.array([[1,1,1], [0,0,0], [-1,-1,-1]], dtype=float) # 가로 엣지
blur = np.ones((3,3), dtype=float) / 9.0

# 십자 모양 테스트 이미지 
cross = np.zeros((9, 9), dtype=float)
cross[4, :] = 1.0 # 가로줄
cross[:, 4] = 1.0 # 세로줄

for name, k in [('세로 엣지', vertical), ('가로 엣지', horizontal), ('블러', blur)]:
    r = conv2d_numpy(cross, k, stride=1, padding=0)
    print(f'--- {name} 커널 반응 (max|val|={np.max(np.abs(r)):2f}) ---')
    print(np.round(r, 1))
    print()
