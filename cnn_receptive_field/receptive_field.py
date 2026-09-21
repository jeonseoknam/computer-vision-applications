def compute_receptive_field(layers):
    r = 1  # r = 현재 Receptive Field 크기
    j = 1  # j = 현재 feature map의 한 칸이 원본 입력에서 몇 칸 간격에 해당하는지(jump)
    # 처음에는 입력의 한 픽셀은 당연히 자기 자신 하나만 보니까 RF=1, jump = 1이다.
    print(f"{'층':<16}{'kernel':>7}{'stride':>7}{'RF':>6}{'jump':>6}")
    print('-' * 42)
    print(f"{'입력':<16}{'-':>7}{r:>6}{j:>6}")
    
    for name, k, s in layers:
        r = r + (k-1) * j
        j = j * s
        print(f'{name:<16}{k:>7}{s:>7}{r:>6}{j:>6}')
    return r, j

print('=== 3x3 conv 5층 (stride=1) ===')
compute_receptive_field([(f'conv{i+1}(3x3)', 3, 1) for i in range(5)])

print()
print('=== conv + 2x2 pool(stride2) 섞기 ===')
compute_receptive_field([
    ('conv1(3x3)', 3, 1),
    ('pool1(2x2)', 2, 2),
    ('conv1(3x3)', 3, 1),
    ('pool2(2x2)', 2, 2),
    ('conv3(3x3)', 3, 1),
])