# Denoising practice

My custom repository for multiple Denoising task.

- inlcude supervised training
- include self-supervised training

- include various dataset
    - SIDD
    - DND

## How to use

## Add custom code

## Results

## Containing functions

---

### development schedule (inverse time line)

    
- basic modules for deep-learning.
    - [ ] Add summary printing module
    - [ ] Save configuration file and network code
    - [ ] Add tensor board output
    - [ ] Add code for printing something important while training
    - [ ] Add binary data conversion
    - [ ] Shape progress module function like tqdm
    - [ ] Add progress bar in progress module
    - [ ] Add cfg change with argument
    - [x] Add test code
    - [x] Use "start_epoch" and "interval_epoch"
    - [x] Add validation image out
    - [x] Add training process for self-supervised manner

### 추가적으로 만들고 싶은 것.

- training을 시킬 때, model code도 같이 저장. (+ 나중에 test할 때 이걸로도 하는 방법.)

### RTCV final project tasks

- verifying real-world noise is real laplace distribution.
- DBSN reproduction.
- applying diverse noise level in single image.
- applying laplace distribution.
- applying fusion technique.
- new blind-spot network architecture.

<!--

### Notepad

- sRGB 이미지가 학습이 안 되는 이유는 주변 픽셀(inner 2 pixel range)들로부터 demosaic function을 학습하여 원본 pixel을 유추할 수 있기 때문일 것이다.
    -> 그러면 spatial한 noise는 되는 가?
- 그러면 처음 center masked conv는 초기 feature를 얻기 위함인데, 이것을 feature로 사용하여 conditioned로 넣어주면 유추는 할 수 없으면서 feature는 넘길 수 있지 않을끼?
- conditional branch로 학습이 안 되게 막는 것은 얻고난 feature 쪽으로 grad가 못하게 detach()하면 될듯.

- DBSN을 그냥 SIDD_val에서 학습.
- RBSN이 얼마나 바뀌는지 확인
- laplacian으로 가정하면 얼마나 더 좋아지는지 확인
- 더 좋아졌을 경우 DND에서 test하여 결과 확인.


- DBSN의 장점은 4방향으로 퍼진다는 것이다.
- EBSN의 장점은 3x3conv를 사용하여 connection이 많고 skip connection이 있는 network를 구성할 수 있다는 것이다.
- 이 둘의 장점을 합칠 수 있을 것이다.
- receptive field가 좁아져도 성능이 늘어나는가?
    - 1x1 conv만 있어도 괜찮은 가를 실험 or paper survey
- near pixel attention을 적용할 방법을 생각.
- training scheme을 Laine or DBSN처럼 하는 방법을 생각. 

-->
